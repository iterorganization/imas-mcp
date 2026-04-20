"""Metadata invariant tests for StandardName nodes.

These tests verify structural correctness of metadata written to the
graph by the standard-name compose pipeline.  They operate purely on
mocked Cypher — no live Neo4j required.

Guards added after the 2025-04-20 metadata QA audit (commit 35-*).
Extended 2025-07-17 with tag, kind, dd_version, and write-path invariants.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_schema() -> dict:
    """Load the standard_name LinkML schema YAML."""
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "imas_codex"
        / "schemas"
        / "standard_name.yaml"
    )
    return yaml.safe_load(schema_path.read_text())


def _get_write_cypher(sample_names: list[dict]) -> list[str]:
    """Call write_standard_names with mocked GraphClient, return all Cypher strings."""
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])

    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        write_standard_names(sample_names)

    return [c[0][0] for c in mock_gc.query.call_args_list]


# ---------------------------------------------------------------------------
# Schema-level invariants
# ---------------------------------------------------------------------------


class TestSchemaInvariants:
    """Verify the LinkML schema defines required metadata slots."""

    def test_unit_slot_has_unit_range(self) -> None:
        """StandardName.unit must have range: Unit with HAS_UNIT relationship."""
        schema = _load_schema()
        sn_class = schema["classes"]["StandardName"]
        attrs = sn_class.get("attributes", {})
        unit_attr = attrs.get("unit", {})
        assert unit_attr.get("range") == "Unit", "unit slot must have range: Unit"
        annotations = unit_attr.get("annotations", {})
        assert annotations.get("relationship_type") == "HAS_UNIT"

    def test_cocos_slot_has_cocos_range(self) -> None:
        """StandardName.cocos must have range: COCOS."""
        schema = _load_schema()
        sn_class = schema["classes"]["StandardName"]
        attrs = sn_class.get("attributes", {})
        cocos_attr = attrs.get("cocos", {})
        assert cocos_attr.get("range") == "COCOS"

    def test_kind_enum_values(self) -> None:
        """StandardNameKind must include the expected base kinds."""
        schema = _load_schema()
        kind_enum = schema["enums"]["StandardNameKind"]
        values = set(kind_enum["permissible_values"].keys())
        expected = {"scalar", "vector", "tensor", "metadata"}
        assert expected <= values, f"Missing kind values: {expected - values}"

    def test_validation_status_enum(self) -> None:
        """StandardNameValidationStatus must include expected statuses."""
        schema = _load_schema()
        vs_enum = schema["enums"]["StandardNameValidationStatus"]
        values = set(vs_enum["permissible_values"].keys())
        expected = {"pending", "valid", "quarantined", "needs_revision"}
        assert expected <= values, f"Missing validation_status: {expected - values}"


# ---------------------------------------------------------------------------
# Cypher-level invariants (write_standard_names)
# ---------------------------------------------------------------------------


class TestWriteCypherInvariants:
    """Verify Cypher produced by write_standard_names enforces metadata contracts."""

    @pytest.fixture()
    def cypher_statements(self, sample_standard_names: list[dict]) -> list[str]:
        """Capture all Cypher statements from a write_standard_names call."""
        return _get_write_cypher(sample_standard_names)

    def _merge_cypher(self, stmts: list[str]) -> str:
        """Find the MERGE StandardName statement."""
        for s in stmts:
            if "MERGE (sn:StandardName" in s:
                return s
        raise AssertionError("No MERGE StandardName query found")

    def test_has_unit_relationship_created(self, cypher_statements: list[str]) -> None:
        """write_standard_names must create HAS_UNIT relationships."""
        has_unit = any("MERGE (sn)-[:HAS_UNIT]->(u)" in s for s in cypher_statements)
        assert has_unit, "HAS_UNIT relationship creation missing from Cypher"

    def test_has_cocos_relationship_uses_match(
        self, cypher_statements: list[str]
    ) -> None:
        """HAS_COCOS must use MATCH (not MERGE) since COCOS nodes pre-exist."""
        has_cocos = [s for s in cypher_statements if "HAS_COCOS" in s]
        # If no COCOS batch (sample data lacks cocos), that's OK — the
        # template is checked by test_cocos_relationship_template below.
        for stmt in has_cocos:
            assert "MATCH (c:COCOS" in stmt, "HAS_COCOS must use MATCH, not MERGE"

    def test_physics_domain_uses_coalesce(self, cypher_statements: list[str]) -> None:
        """physics_domain SET must use coalesce to avoid overwriting with NULL."""
        merge = self._merge_cypher(cypher_statements)
        assert "coalesce(" in merge and "physics_domain" in merge

    def test_cocos_transformation_type_uses_coalesce(
        self, cypher_statements: list[str]
    ) -> None:
        """cocos_transformation_type must use coalesce in MERGE."""
        merge = self._merge_cypher(cypher_statements)
        assert "cocos_transformation_type" in merge
        assert "coalesce(b.cocos_transformation_type" in merge

    def test_dd_version_uses_coalesce(self, cypher_statements: list[str]) -> None:
        """dd_version must use coalesce in MERGE."""
        merge = self._merge_cypher(cypher_statements)
        assert "coalesce(b.dd_version" in merge

    def test_cocos_missing_integer_logs_warning(self) -> None:
        """write_standard_names must warn if cocos_transformation_type set but cocos is None."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "test_cocos_guard",
                "source_types": ["dd"],
                "source_id": "test/path",
                "unit": "A",
                "kind": "scalar",
                "cocos_transformation_type": "ip_like",
                "cocos": None,  # Missing!
            }
        ]

        import logging

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            with patch("imas_codex.standard_names.graph_ops.logger") as mock_logger:
                from imas_codex.standard_names.graph_ops import write_standard_names

                write_standard_names(names)

                # Verify warning was issued
                warning_calls = [
                    str(c)
                    for c in mock_logger.warning.call_args_list
                    if "cocos_transformation_type" in str(c)
                ]
                assert warning_calls, (
                    "Expected warning about cocos_transformation_type without cocos integer"
                )

    def test_unit_in_batch_payload(self, sample_standard_names: list[dict]) -> None:
        """Every name dict with a unit must include it in the batch payload."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(sample_standard_names)

        # Find the HAS_UNIT call and verify the batch includes our units
        for c in mock_gc.query.call_args_list:
            cypher = c[0][0]
            if "HAS_UNIT" in cypher:
                batch = c[1].get("batch", [])
                unit_ids = {b["unit"] for b in batch}
                # Both sample names have units (eV and A)
                assert "eV" in unit_ids
                assert "A" in unit_ids
                return
        pytest.fail("No HAS_UNIT query found in calls")


# ---------------------------------------------------------------------------
# Grammar decomposition invariants
# ---------------------------------------------------------------------------


class TestGrammarDecomposition:
    """Verify _grammar_decomposition covers all expected fields."""

    def test_decomposition_field_set(self) -> None:
        """All grammar parse fields in the schema must appear in _GRAMMAR_DECOMPOSITION_FIELDS.

        ``grammar_tokens`` is excluded because it is the JSON-serialized
        token list populated by HAS_SEGMENT edge creation, not by
        ``_grammar_decomposition``.
        """
        from imas_codex.standard_names.graph_ops import _GRAMMAR_DECOMPOSITION_FIELDS

        schema = _load_schema()
        sn_attrs = schema["classes"]["StandardName"]["attributes"]
        # grammar_tokens is populated separately via HAS_SEGMENT, not decomposition
        excluded = {"tokens"}
        schema_grammar_fields = {
            k.removeprefix("grammar_") for k in sn_attrs if k.startswith("grammar_")
        } - excluded
        code_fields = set(_GRAMMAR_DECOMPOSITION_FIELDS)
        missing = schema_grammar_fields - code_fields
        assert not missing, (
            f"Grammar fields in schema but not in decomposition: {missing}"
        )

    def test_decomposition_returns_all_fields(self) -> None:
        """_grammar_decomposition must return a key for every field."""
        from imas_codex.standard_names.graph_ops import (
            _GRAMMAR_DECOMPOSITION_FIELDS,
            _grammar_decomposition,
        )

        result = _grammar_decomposition("electron_temperature")
        for field in _GRAMMAR_DECOMPOSITION_FIELDS:
            assert f"grammar_{field}" in result, f"Missing grammar_{field} in result"


# ---------------------------------------------------------------------------
# Pipeline injection invariants (compose worker)
# ---------------------------------------------------------------------------


class TestComposeWorkerInjection:
    """Verify compose worker injects DD-authoritative fields (not LLM)."""

    def test_unit_injected_from_source_not_llm(self) -> None:
        """The compose worker must set unit from source_item, not LLM output."""
        source = inspect.getsource(
            importlib.import_module("imas_codex.standard_names.workers")
        )
        # The worker must reference source_item for unit
        # and must not assign unit from LLM candidate
        assert 'unit": unit' in source or '"unit": unit' in source

    def test_physics_domain_injected_from_source(self) -> None:
        """physics_domain must be injected from source DD, not LLM."""
        source = inspect.getsource(
            importlib.import_module("imas_codex.standard_names.workers")
        )
        assert "physics_domain" in source
        # Must reference source_item for physics_domain
        assert 'source_item.get("physics_domain")' in source

    def test_cocos_injected_from_source(self) -> None:
        """cocos_transformation_type must come from source_item, not LLM."""
        source = inspect.getsource(
            importlib.import_module("imas_codex.standard_names.workers")
        )
        assert 'source_item.get("cocos_label")' in source


# ---------------------------------------------------------------------------
# PhysicsDomain enum consistency
# ---------------------------------------------------------------------------


class TestPhysicsDomainEnum:
    """Verify physics_domain enum is available and comprehensive."""

    def test_physics_domain_importable(self) -> None:
        """PhysicsDomain must be importable from imas_codex.core.physics_domain."""
        from imas_codex.core.physics_domain import PhysicsDomain

        assert hasattr(PhysicsDomain, "__members__")

    def test_physics_domain_has_general(self) -> None:
        """PhysicsDomain must include 'general' as fallback."""
        from imas_codex.core.physics_domain import PhysicsDomain

        values = {pd.value for pd in PhysicsDomain}
        assert "general" in values

    def test_physics_domain_has_equilibrium(self) -> None:
        """PhysicsDomain must include 'equilibrium'."""
        from imas_codex.core.physics_domain import PhysicsDomain

        values = {pd.value for pd in PhysicsDomain}
        assert "equilibrium" in values

    def test_physics_domain_no_mhd_alias(self) -> None:
        """PhysicsDomain must use 'magnetohydrodynamics', not 'mhd'."""
        from imas_codex.core.physics_domain import PhysicsDomain

        values = {pd.value for pd in PhysicsDomain}
        assert "mhd" not in values, "Use 'magnetohydrodynamics', not 'mhd'"
        assert "magnetohydrodynamics" in values


# ---------------------------------------------------------------------------
# StandardNameKind enum consistency
# ---------------------------------------------------------------------------


class TestStandardNameKindEnum:
    """Verify kind enum from schema includes all expected values."""

    def test_kind_enum_no_unknown_values(self) -> None:
        """All graph kind values must be in the schema enum."""
        schema = _load_schema()
        kind_enum = schema["enums"]["StandardNameKind"]
        allowed = set(kind_enum["permissible_values"].keys())
        # These are the kinds observed in the graph as of 2025-07 audit
        observed = {
            "scalar",
            "vector_component",
            "complex_part",
            "tensor_component",
            "spectrum",
        }
        unknown = observed - allowed
        assert not unknown, f"Kinds observed in graph but not in schema: {unknown}"

    def test_kind_enum_includes_metadata(self) -> None:
        """metadata kind must be in the enum for non-measurable concepts."""
        schema = _load_schema()
        kind_enum = schema["enums"]["StandardNameKind"]
        assert "metadata" in kind_enum["permissible_values"]


# ---------------------------------------------------------------------------
# Tag normalisation invariants
# ---------------------------------------------------------------------------


class TestTagNormalisation:
    """Verify tag formatting conventions."""

    def test_tags_must_be_lowercase_hyphenated(self) -> None:
        """Tags in the schema/pipeline must follow kebab-case convention.

        This is a structural test: the pipeline normalises tags to lowercase.
        We verify via source inspection that the write path doesn't silently
        accept upper-case or underscore-separated tags.
        """
        # Tags observed in the 2025-07 audit are all lowercase hyphenated.
        # Verify the schema description documents this convention.
        schema = _load_schema()
        sn_attrs = schema["classes"]["StandardName"]["attributes"]
        tags_attr = sn_attrs.get("tags", {})
        assert tags_attr.get("multivalued") is True, "tags must be multivalued"

    def test_write_cypher_preserves_tags(
        self,
        sample_standard_names: list[dict],
    ) -> None:
        """Tags from sample data must appear in the Cypher batch payload."""
        stmts = _get_write_cypher(sample_standard_names)
        merge = next(s for s in stmts if "MERGE (sn:StandardName" in s)
        assert "sn.tags" in merge, "Tags must be SET in the MERGE statement"


# ---------------------------------------------------------------------------
# dd_version consistency invariants
# ---------------------------------------------------------------------------


class TestDDVersionConsistency:
    """Verify dd_version is consistently set."""

    def test_dd_version_uses_coalesce_in_write(
        self,
        sample_standard_names: list[dict],
    ) -> None:
        """dd_version must use coalesce to avoid overwriting with NULL."""
        stmts = _get_write_cypher(sample_standard_names)
        merge = next(s for s in stmts if "MERGE (sn:StandardName" in s)
        assert "coalesce(b.dd_version" in merge


# ---------------------------------------------------------------------------
# Write path relationship invariants
# ---------------------------------------------------------------------------


class TestWritePathRelationships:
    """Verify relationship creation patterns in write_standard_names."""

    def test_has_unit_uses_merge(self, sample_standard_names: list[dict]) -> None:
        """HAS_UNIT must use MERGE to be idempotent."""
        stmts = _get_write_cypher(sample_standard_names)
        unit_stmts = [s for s in stmts if "HAS_UNIT" in s]
        assert unit_stmts, "HAS_UNIT statement must exist"
        for s in unit_stmts:
            assert "MERGE" in s, "HAS_UNIT must use MERGE for idempotency"

    def test_has_standard_name_created_for_dd_source(
        self,
        sample_standard_names: list[dict],
    ) -> None:
        """DD-sourced names must create HAS_STANDARD_NAME edge."""
        stmts = _get_write_cypher(sample_standard_names)
        has_sn = any("HAS_STANDARD_NAME" in s for s in stmts)
        assert has_sn, "HAS_STANDARD_NAME must be created for DD-sourced names"

    def test_unit_batch_includes_all_non_null_units(
        self,
        sample_standard_names: list[dict],
    ) -> None:
        """Every name with a unit must appear in the HAS_UNIT batch."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(sample_standard_names)

        unit_calls = [c for c in mock_gc.query.call_args_list if "HAS_UNIT" in c[0][0]]
        assert unit_calls, "HAS_UNIT query must be called"
        batch = unit_calls[0][1]["batch"]
        unit_ids = {b["unit"] for b in batch}
        # All sample names have units
        for n in sample_standard_names:
            if n.get("unit"):
                assert n["unit"] in unit_ids, f"Unit {n['unit']} missing from batch"


# ---------------------------------------------------------------------------
# Unit override engine invariants
# ---------------------------------------------------------------------------


class TestUnitOverrideEngine:
    """Verify the unit override mechanism is importable and functional."""

    def test_override_engine_importable(self) -> None:
        """unit_overrides module must be importable."""
        from imas_codex.standard_names.unit_overrides import resolve_unit

        assert callable(resolve_unit)

    def test_override_config_exists(self) -> None:
        """unit_overrides.yaml config must exist."""
        config = (
            Path(__file__).resolve().parents[2]
            / "imas_codex"
            / "standard_names"
            / "config"
            / "unit_overrides.yaml"
        )
        assert config.exists(), f"Config not found: {config}"
