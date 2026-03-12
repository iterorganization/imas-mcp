"""End-to-end tests for the IMAS mapping pipeline.

Tests the full pipeline chain: tools → models → orchestrator → CLI,
using mocked GraphClient and LLM calls to validate the integration
without requiring a live Neo4j database or LLM API.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.ids.models import (
    EscalationFlag,
    EscalationSeverity,
    FieldMappingBatch,
    FieldMappingEntry,
    SectionAssignment,
    SectionAssignmentBatch,
    ValidatedFieldMapping,
    ValidatedMappingResult,
    persist_mapping_result,
)
from imas_codex.ids.tools import (
    analyze_units,
    check_imas_paths,
    fetch_imas_fields,
    fetch_imas_subtree,
    get_sign_flip_paths,
    query_signal_sources,
    search_existing_mappings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gc():
    """Create a mock GraphClient."""
    gc = MagicMock()
    gc.query.return_value = []
    return gc


@pytest.fixture
def sample_groups():
    """Sample signal groups from a JET pf_active scenario."""
    return [
        {
            "id": "jet:pf_coils:group1",
            "group_key": "pf_coil_1",
            "description": "PF coil 1 position and geometry",
            "keywords": ["pf_active", "coil"],
            "physics_domain": "magnetic_field_systems",
            "status": "enriched",
            "member_count": 6,
            "sample_members": [
                "jet:device_xml:p68613:pfcoils:1:r",
                "jet:device_xml:p68613:pfcoils:1:z",
            ],
            "sample_accessors": [
                "\\ppf::pfcoils:1:r",
                "\\ppf::pfcoils:1:z",
            ],
            "rep_description": "PF coil 1 radial position in metres",
            "rep_unit": "m",
            "rep_sign_convention": None,
            "rep_cocos": None,
            "imas_mappings": [],
        },
        {
            "id": "jet:pf_coils:group2",
            "group_key": "pf_coil_2",
            "description": "PF coil 2 position and geometry",
            "keywords": ["pf_active", "coil"],
            "physics_domain": "magnetic_field_systems",
            "status": "enriched",
            "member_count": 6,
            "sample_members": [
                "jet:device_xml:p68613:pfcoils:2:r",
                "jet:device_xml:p68613:pfcoils:2:z",
            ],
            "sample_accessors": [
                "\\ppf::pfcoils:2:r",
                "\\ppf::pfcoils:2:z",
            ],
            "rep_description": "PF coil 2 radial position in metres",
            "rep_unit": "m",
            "rep_sign_convention": None,
            "rep_cocos": None,
            "imas_mappings": [],
        },
    ]


@pytest.fixture
def sample_subtree():
    """Sample IMAS subtree for pf_active."""
    return [
        {
            "id": "pf_active/coil",
            "name": "coil",
            "data_type": "STRUCT_ARRAY",
            "node_type": "static",
            "documentation": "PF coil descriptions",
            "units": None,
        },
        {
            "id": "pf_active/coil/element/geometry/rectangle/r",
            "name": "r",
            "data_type": "FLT_0D",
            "node_type": "static",
            "documentation": "Geometric centre R",
            "units": "m",
        },
        {
            "id": "pf_active/coil/element/geometry/rectangle/z",
            "name": "z",
            "data_type": "FLT_0D",
            "node_type": "static",
            "documentation": "Geometric centre Z",
            "units": "m",
        },
    ]


@pytest.fixture
def sample_section_assignment():
    """Sample section assignment result from Step 1."""
    return SectionAssignmentBatch(
        ids_name="pf_active",
        assignments=[
            SectionAssignment(
                source_id="jet:pf_coils:group1",
                imas_section_path="pf_active/coil",
                confidence=0.95,
                reasoning="PF coil geometry maps to pf_active/coil",
            ),
            SectionAssignment(
                source_id="jet:pf_coils:group2",
                imas_section_path="pf_active/coil",
                confidence=0.90,
                reasoning="PF coil 2 geometry maps to pf_active/coil",
            ),
        ],
        unassigned_groups=[],
    )


@pytest.fixture
def sample_field_batch():
    """Sample field mapping batch from Step 2."""
    return FieldMappingBatch(
        ids_name="pf_active",
        section_path="pf_active/coil",
        mappings=[
            FieldMappingEntry(
                source_id="jet:pf_coils:group1",
                source_property="value",
                target_id="pf_active/coil/element/geometry/rectangle/r",
                transform_expression="value",
                source_units="m",
                target_units="m",
                confidence=0.95,
                reasoning="Direct R position mapping",
            ),
            FieldMappingEntry(
                source_id="jet:pf_coils:group1",
                source_property="value",
                target_id="pf_active/coil/element/geometry/rectangle/z",
                transform_expression="value",
                source_units="m",
                target_units="m",
                confidence=0.90,
                reasoning="Direct Z position mapping",
            ),
        ],
        escalations=[
            EscalationFlag(
                source_id="jet:pf_coils:group2",
                target_id="pf_active/coil/current/data",
                severity=EscalationSeverity.WARNING,
                reason="Current signal has ambiguous units",
            ),
        ],
    )


@pytest.fixture
def sample_validated_result():
    """Sample validated mapping result from Step 3."""
    return ValidatedMappingResult(
        facility="jet",
        ids_name="pf_active",
        dd_version="4.1.1",
        sections=[
            SectionAssignment(
                source_id="jet:pf_coils:group1",
                imas_section_path="pf_active/coil",
                confidence=0.95,
                reasoning="PF coil geometry maps to pf_active/coil",
            ),
        ],
        bindings=[
            ValidatedFieldMapping(
                source_id="jet:pf_coils:group1",
                source_property="value",
                target_id="pf_active/coil/element/geometry/rectangle/r",
                transform_expression="value",
                source_units="m",
                target_units="m",
                confidence=0.95,
            ),
            ValidatedFieldMapping(
                source_id="jet:pf_coils:group1",
                source_property="value",
                target_id="pf_active/coil/element/geometry/rectangle/z",
                transform_expression="value",
                source_units="m",
                target_units="m",
                confidence=0.90,
            ),
        ],
        escalations=[
            EscalationFlag(
                source_id="jet:pf_coils:group2",
                target_id="pf_active/coil/current/data",
                severity=EscalationSeverity.WARNING,
                reason="Current signal has ambiguous units",
            ),
        ],
        corrections=["Updated coil R path from deprecated alias"],
    )


# ---------------------------------------------------------------------------
# Tool function tests
# ---------------------------------------------------------------------------


class TestFetchImasSubtree:
    def test_returns_subtree(self, mock_gc, sample_subtree):
        mock_gc.query.return_value = sample_subtree
        result = fetch_imas_subtree("pf_active", gc=mock_gc)
        assert len(result) == 3
        assert result[0]["id"] == "pf_active/coil"

    def test_with_path_prefix(self, mock_gc, sample_subtree):
        mock_gc.query.return_value = sample_subtree[1:]
        result = fetch_imas_subtree("pf_active", "coil", gc=mock_gc, leaf_only=True)
        assert len(result) == 2

    def test_empty_ids(self, mock_gc):
        mock_gc.query.return_value = []
        result = fetch_imas_subtree("nonexistent", gc=mock_gc)
        assert result == []


class TestFetchImasFields:
    def test_returns_field_details(self, mock_gc):
        mock_gc.query.return_value = [
            {
                "id": "pf_active/coil/element/geometry/rectangle/r",
                "name": "r",
                "ids": "pf_active",
                "documentation": "Geometric centre R",
                "data_type": "FLT_0D",
                "node_type": "static",
                "ndim": 0,
                "physics_domain": "magnetic_field_systems",
                "units": "m",
                "cluster_labels": [],
                "coordinates": [],
            }
        ]
        result = fetch_imas_fields(
            "pf_active",
            ["coil/element/geometry/rectangle/r"],
            gc=mock_gc,
        )
        assert len(result) == 1
        assert result[0]["units"] == "m"


class TestCheckImasPaths:
    def test_valid_path(self, mock_gc):
        mock_gc.query.return_value = [
            {"id": "pf_active/coil", "data_type": "STRUCT_ARRAY", "units": None}
        ]
        result = check_imas_paths(["pf_active/coil"], gc=mock_gc)
        assert result[0]["exists"] is True

    def test_invalid_path_no_rename(self, mock_gc):
        mock_gc.query.side_effect = [[], []]  # main query, rename query
        result = check_imas_paths(["nonexistent/path"], gc=mock_gc)
        assert result[0]["exists"] is False
        assert "suggestion" not in result[0]

    def test_renamed_path(self, mock_gc):
        mock_gc.query.side_effect = [
            [],  # main query — not found
            [{"new_path": "pf_active/coil"}],  # rename query
        ]
        result = check_imas_paths(["old/path"], gc=mock_gc)
        assert result[0]["exists"] is False
        assert result[0]["suggestion"] == "pf_active/coil"


class TestQuerySignalSources:
    def test_returns_groups(self, mock_gc, sample_groups):
        mock_gc.query.return_value = sample_groups
        result = query_signal_sources("jet", gc=mock_gc)
        assert len(result) == 2
        assert result[0]["group_key"] == "pf_coil_1"

    def test_filtered_by_ids(self, mock_gc, sample_groups):
        mock_gc.query.return_value = sample_groups[:1]
        result = query_signal_sources("jet", "pf_active", gc=mock_gc)
        assert len(result) == 1


class TestSearchExistingMappings:
    def test_no_existing_mapping(self, mock_gc):
        mock_gc.query.return_value = []
        result = search_existing_mappings("jet", "pf_active", gc=mock_gc)
        assert result["mapping"] is None
        assert result["sections"] == []
        assert result["bindings"] == []

    def test_existing_mapping(self, mock_gc):
        mock_gc.query.side_effect = [
            [
                {
                    "id": "jet:pf_active",
                    "facility_id": "jet",
                    "ids_name": "pf_active",
                    "dd_version": "4.1.1",
                    "status": "validated",
                    "provider": "imas-codex",
                }
            ],
            [
                {
                    "imas_path": "pf_active/coil",
                    "data_type": "STRUCT_ARRAY",
                    "config": "{}",
                }
            ],
            [
                {
                    "source_id": "jet:pf_coils:group1",
                    "target_id": "pf_active/coil/element/geometry/rectangle/r",
                    "transform_expression": "value",
                    "source_units": "m",
                    "target_units": "m",
                    "source_property": "value",
                }
            ],
        ]
        result = search_existing_mappings("jet", "pf_active", gc=mock_gc)
        assert result["mapping"]["id"] == "jet:pf_active"
        assert len(result["sections"]) == 1
        assert len(result["bindings"]) == 1


class TestAnalyzeUnits:
    def test_compatible_units(self):
        result = analyze_units("m", "m")
        assert result["compatible"] is True
        assert result["conversion_factor"] == pytest.approx(1.0)

    def test_none_units(self):
        result = analyze_units(None, None)
        assert result["compatible"] is True

    def test_missing_one_unit(self):
        result = analyze_units("m", None)
        assert result["compatible"] is False


class TestGetSignFlipPaths:
    def test_returns_list(self):
        # May return empty list if imas-python not installed with sign flip data
        result = get_sign_flip_paths("pf_active")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------


class TestSectionAssignmentBatch:
    def test_serialization(self, sample_section_assignment):
        data = sample_section_assignment.model_dump()
        restored = SectionAssignmentBatch.model_validate(data)
        assert len(restored.assignments) == 2
        assert restored.ids_name == "pf_active"

    def test_json_roundtrip(self, sample_section_assignment):
        json_str = sample_section_assignment.model_dump_json()
        restored = SectionAssignmentBatch.model_validate_json(json_str)
        assert restored.assignments[0].confidence == 0.95


class TestFieldMappingBatch:
    def test_serialization(self, sample_field_batch):
        data = sample_field_batch.model_dump()
        restored = FieldMappingBatch.model_validate(data)
        assert len(restored.mappings) == 2
        assert len(restored.escalations) == 1

    def test_escalation_severity(self, sample_field_batch):
        esc = sample_field_batch.escalations[0]
        assert esc.severity == EscalationSeverity.WARNING
        assert esc.severity.value == "warning"


class TestValidatedMappingResult:
    def test_serialization(self, sample_validated_result):
        data = sample_validated_result.model_dump()
        restored = ValidatedMappingResult.model_validate(data)
        assert len(restored.bindings) == 2
        assert len(restored.escalations) == 1
        assert len(restored.corrections) == 1

    def test_json_roundtrip(self, sample_validated_result):
        json_str = sample_validated_result.model_dump_json()
        restored = ValidatedMappingResult.model_validate_json(json_str)
        assert restored.facility == "jet"
        assert restored.dd_version == "4.1.1"


class TestPersistMappingResult:
    def test_creates_mapping_node(self, mock_gc, sample_validated_result):
        persist_mapping_result(sample_validated_result, gc=mock_gc)
        # Should have called query multiple times:
        # 1 for MERGE mapping, 1 for POPULATES, 1 for USES_SIGNAL_SOURCE,
        # 2 for MAPS_TO_IMAS (2 fields), 1 for escalation evidence
        assert mock_gc.query.call_count == 6

    def test_returns_mapping_id(self, mock_gc, sample_validated_result):
        result = persist_mapping_result(sample_validated_result, gc=mock_gc)
        assert result == "jet:pf_active"

    def test_escalation_persisted(self, mock_gc, sample_validated_result):
        persist_mapping_result(sample_validated_result, gc=mock_gc)
        # Find the escalation call (last one)
        calls = mock_gc.query.call_args_list
        evidence_call = calls[-1]
        assert "MappingEvidence" in evidence_call[0][0]
        assert evidence_call[1]["severity"] == "warning"

    def test_default_status_is_generated(self, mock_gc, sample_validated_result):
        persist_mapping_result(sample_validated_result, gc=mock_gc)
        # First call is the MERGE IMASMapping — check the status param
        merge_call = mock_gc.query.call_args_list[0]
        assert merge_call[1]["status"] == "generated"

    def test_custom_status(self, mock_gc, sample_validated_result):
        persist_mapping_result(
            sample_validated_result, gc=mock_gc, status="active"
        )
        merge_call = mock_gc.query.call_args_list[0]
        assert merge_call[1]["status"] == "active"


# ---------------------------------------------------------------------------
# Pipeline orchestrator tests
# ---------------------------------------------------------------------------


class TestPipelineOrchestrator:
    """Test the full pipeline with mocked LLM and graph."""

    @patch("imas_codex.ids.mapping.search_imas_semantic")
    def test_gather_context(
        self, mock_semantic, mock_gc, sample_groups, sample_subtree
    ):
        """Test context gathering."""
        from imas_codex.ids.mapping import gather_context

        mock_semantic.return_value = sample_subtree

        # Patch fetch_imas_subtree to use our mock
        with (
            patch("imas_codex.ids.mapping.fetch_imas_subtree") as mock_subtree,
            patch("imas_codex.ids.mapping.query_signal_sources") as mock_qsg,
            patch("imas_codex.ids.mapping.search_existing_mappings") as mock_sem,
        ):
            mock_subtree.return_value = sample_subtree
            mock_qsg.return_value = sample_groups
            mock_sem.return_value = {
                "mapping": None,
                "sections": [],
                "bindings": [],
            }

            ctx = gather_context("jet", "pf_active", gc=mock_gc)

        assert len(ctx["groups"]) == 2
        assert len(ctx["subtree"]) == 3

    @patch("imas_codex.ids.mapping._call_llm")
    def test_assign_sections(
        self, mock_call_llm, sample_groups, sample_subtree, sample_section_assignment
    ):
        """Test section assignment."""
        from imas_codex.ids.mapping import PipelineCost, assign_sections

        mock_call_llm.return_value = sample_section_assignment
        cost = PipelineCost()

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "semantic": sample_subtree,
        }

        result = assign_sections("jet", "pf_active", context, cost=cost)
        assert len(result.assignments) == 2
        mock_call_llm.assert_called_once()

    @patch("imas_codex.ids.mapping.fetch_source_code_refs")
    @patch("imas_codex.ids.mapping._call_llm")
    @patch("imas_codex.ids.mapping.fetch_imas_fields")
    @patch("imas_codex.ids.mapping.fetch_imas_subtree")
    def test_map_signals(
        self,
        mock_subtree,
        mock_fields,
        mock_call_llm,
        mock_code_refs,
        mock_gc,
        sample_groups,
        sample_subtree,
        sample_section_assignment,
        sample_field_batch,
    ):
        """Test signal mapping generation."""
        from imas_codex.ids.mapping import PipelineCost, map_signals

        mock_fields.return_value = sample_subtree[1:]
        mock_subtree.return_value = sample_subtree[1:]
        mock_call_llm.return_value = sample_field_batch
        mock_code_refs.return_value = []
        cost = PipelineCost()

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "cocos_paths": [],
            "existing": {"mapping": None, "sections": [], "bindings": []},
        }

        result = map_signals(
            "jet",
            "pf_active",
            sample_section_assignment,
            context,
            gc=mock_gc,
            cost=cost,
        )
        # Called once per section assignment (2 assignments → same section)
        assert len(result) == 2
        assert len(result[0].mappings) == 2

    @patch("imas_codex.ids.validation.check_imas_paths")
    def test_validate_mappings(
        self,
        mock_check,
        mock_gc,
        sample_section_assignment,
        sample_field_batch,
    ):
        """Test validation — programmatic, not LLM."""
        from imas_codex.ids.mapping import validate_mappings

        mock_check.return_value = [
            {"path": "pf_active/coil/element/geometry/rectangle/r", "exists": True},
            {"path": "pf_active/coil/element/geometry/rectangle/z", "exists": True},
        ]
        # Source exists query
        mock_gc.query.return_value = [{"id": "some_group"}]

        result = validate_mappings(
            "jet",
            "pf_active",
            "4.1.1",
            sample_section_assignment,
            [sample_field_batch],
            gc=mock_gc,
        )
        assert len(result.bindings) == 2
        assert result.facility == "jet"
        assert result.dd_version == "4.1.1"

    @patch("imas_codex.ids.mapping.validate_mappings")
    @patch("imas_codex.ids.mapping.discover_assembly")
    @patch("imas_codex.ids.mapping.map_signals")
    @patch("imas_codex.ids.mapping.assign_sections")
    @patch("imas_codex.ids.mapping.gather_context")
    def test_generate_mapping_full_pipeline(
        self,
        mock_step0,
        mock_step1,
        mock_step2,
        mock_step3,
        mock_step4,
        mock_gc,
        sample_groups,
        sample_subtree,
        sample_section_assignment,
        sample_field_batch,
        sample_validated_result,
    ):
        """Test full pipeline end-to-end."""
        from imas_codex.ids.mapping import generate_mapping

        mock_step0.return_value = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "semantic": sample_subtree,
            "existing": {"mapping": None, "sections": [], "bindings": []},
            "cocos_paths": [],
        }
        mock_step1.return_value = sample_section_assignment
        mock_step2.return_value = [sample_field_batch]
        mock_step3.return_value = None  # No assembly patterns discovered
        mock_step4.return_value = sample_validated_result

        result = generate_mapping(
            "jet",
            "pf_active",
            dd_version="4.1.1",
            persist=False,
            gc=mock_gc,
        )

        assert result.mapping_id == "jet:pf_active"
        assert len(result.validated.bindings) == 2
        assert len(result.validated.escalations) == 1
        assert result.persisted is False
        assert result.unassigned_groups == []

    @patch("imas_codex.ids.mapping.validate_mappings")
    @patch("imas_codex.ids.mapping.discover_assembly")
    @patch("imas_codex.ids.mapping.map_signals")
    @patch("imas_codex.ids.mapping.assign_sections")
    @patch("imas_codex.ids.mapping.gather_context")
    def test_generate_mapping_surfaces_unassigned_groups(
        self,
        mock_step0,
        mock_step1,
        mock_step2,
        mock_step3,
        mock_step4,
        mock_gc,
        sample_groups,
        sample_subtree,
        sample_field_batch,
        sample_validated_result,
    ):
        """Test that unassigned_groups from Step 1 are surfaced in MappingResult."""
        from imas_codex.ids.mapping import generate_mapping

        unassigned = SectionAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                SectionAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_section_path="pf_active/coil",
                    confidence=0.95,
                    reasoning="PF coil geometry maps to pf_active/coil",
                ),
            ],
            unassigned_groups=["jet:pf_coils:group3", "jet:pf_coils:group4"],
        )

        mock_step0.return_value = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "semantic": sample_subtree,
            "existing": {"mapping": None, "sections": [], "bindings": []},
            "cocos_paths": [],
        }
        mock_step1.return_value = unassigned
        mock_step2.return_value = [sample_field_batch]
        mock_step3.return_value = None  # No assembly patterns discovered
        mock_step4.return_value = sample_validated_result

        result = generate_mapping(
            "jet",
            "pf_active",
            dd_version="4.1.1",
            persist=False,
            gc=mock_gc,
        )

        assert result.unassigned_groups == [
            "jet:pf_coils:group3",
            "jet:pf_coils:group4",
        ]

    @patch("imas_codex.ids.mapping.gather_context")
    def test_generate_mapping_no_groups_raises(self, mock_step0, mock_gc):
        """Test that empty groups raises ValueError."""
        from imas_codex.ids.mapping import generate_mapping

        mock_step0.return_value = {
            "groups": [],
            "subtree": [],
            "semantic": [],
            "existing": {"mapping": None, "sections": [], "bindings": []},
            "cocos_paths": [],
        }

        with pytest.raises(ValueError, match="No signal sources found"):
            generate_mapping("jet", "pf_active", dd_version="4.1.1", gc=mock_gc)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestMapCLI:
    def test_map_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["--help"])
        assert result.exit_code == 0
        assert "IMAS mapping pipeline" in result.output

    def test_map_status_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show mapping status" in result.output

    def test_map_show_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["show", "--help"])
        assert result.exit_code == 0

    def test_map_validate_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["validate", "--help"])
        assert result.exit_code == 0

    def test_map_clear_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["clear", "--help"])
        assert result.exit_code == 0

    def test_map_activate_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["activate", "--help"])
        assert result.exit_code == 0
        assert "active" in result.output.lower()

    def test_map_no_args_shows_help(self):
        from imas_codex.cli.map import map_cmd

        runner = CliRunner()
        result = runner.invoke(map_cmd, [])
        # Group with no subcommand shows usage
        assert "IMAS mapping pipeline" in result.output or result.exit_code == 0

    @patch("imas_codex.ids.mapping.generate_mapping")
    def test_map_run(self, mock_generate, sample_validated_result):
        from imas_codex.cli.map import map_cmd
        from imas_codex.ids.mapping import MappingResult, PipelineCost

        mock_generate.return_value = MappingResult(
            mapping_id="jet:pf_active",
            validated=sample_validated_result,
            cost=PipelineCost(steps={"step1": 0.001, "step2": 0.002}),
            persisted=True,
            unassigned_groups=["jet:pf_coils:group3"],
            assembly=None,
        )

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["run", "jet", "pf_active", "--no-persist"])
        assert result.exit_code == 0
        assert "jet:pf_active" in result.output
        assert "Bindings: 2" in result.output
        assert "Unassigned signal groups" in result.output
        assert "jet:pf_coils:group3" in result.output

    @patch("imas_codex.ids.mapping.generate_mapping")
    def test_map_run_error(self, mock_generate):
        from imas_codex.cli.map import map_cmd

        mock_generate.side_effect = ValueError("No signal groups found")

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["run", "jet", "pf_active"])
        assert result.exit_code == 1
        assert "No signal groups found" in result.output


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    """Test that prompt templates exist and render correctly."""

    def test_section_assignment_prompt_exists(self):
        from imas_codex.ids.mapping import _load_prompt

        prompt = _load_prompt("section_assignment")
        assert "signal sources" in prompt.lower()
        assert "{{ facility }}" in prompt or "facility" in prompt.lower()

    def test_signal_mapping_prompt_exists(self):
        from imas_codex.ids.mapping import _load_prompt

        prompt = _load_prompt("signal_mapping")
        assert "signal" in prompt.lower()
        assert "transform" in prompt.lower()

    def test_validation_prompt_exists(self):
        from imas_codex.ids.mapping import _load_prompt

        prompt = _load_prompt("validation")
        assert "valid" in prompt.lower()

    def test_section_assignment_prompt_renders(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "section_assignment",
            facility="jet",
            ids_name="pf_active",
            signal_sources="- group1: PF coil 1",
            imas_subtree="pf_active/coil (STRUCT_ARRAY)",
            semantic_results="pf_active/coil — PF coil descriptions",
        )
        assert "jet" in rendered
        assert "pf_active" in rendered
        assert "group1" in rendered

    def test_signal_mapping_prompt_renders(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "signal_mapping",
            facility="jet",
            ids_name="pf_active",
            section_path="pf_active/coil",
            signal_source_detail="{}",
            imas_fields="- pf_active/coil/element/geometry/rectangle/r (FLT_0D) [m]",
            unit_analysis="m → m: compatible",
            cocos_paths="(none)",
            existing_mappings="{}",
            code_references="(no code references available)",
            source_cocos="(no COCOS context)",
        )
        assert "pf_active/coil" in rendered

    def test_signal_mapping_prompt_has_transform_examples(self):
        from imas_codex.ids.mapping import _load_prompt

        prompt = _load_prompt("signal_mapping")
        # Phase 6: transform expression examples
        assert "value * 1e-3" in prompt
        assert "math.radians(value)" in prompt
        assert "convert_units(value" in prompt
        # Phase 6: unit mismatch rule
        assert "source_units" in prompt
        assert "MUST" in prompt

    def test_validation_prompt_renders(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "validation",
            facility="jet",
            ids_name="pf_active",
            dd_version="4.1.1",
            proposed_mappings="[]",
            validation_results="[]",
            existing_mappings="{}",
            escalations="[]",
        )
        assert "jet" in rendered
        assert "4.1.1" in rendered


# ---------------------------------------------------------------------------
# Integration: Pipeline cost tracking
# ---------------------------------------------------------------------------


class TestPipelineCost:
    def test_cost_accumulation(self):
        from imas_codex.ids.mapping import PipelineCost

        cost = PipelineCost()
        cost.add("step1", 0.001, 100)
        cost.add("step2", 0.002, 200)
        cost.add("step2", 0.001, 100)  # Second call to step2

        assert cost.total_usd == pytest.approx(0.004)
        assert cost.total_tokens == 400
        assert cost.steps["step2"] == pytest.approx(0.003)

    def test_empty_cost(self):
        from imas_codex.ids.mapping import PipelineCost

        cost = PipelineCost()
        assert cost.total_usd == 0.0
        assert cost.total_tokens == 0


# ---------------------------------------------------------------------------
# Formatter helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_subtree(self, sample_subtree):
        from imas_codex.ids.mapping import _format_subtree

        result = _format_subtree(sample_subtree)
        assert "pf_active/coil" in result
        assert "STRUCT_ARRAY" in result
        assert "Geometric centre R" in result

    def test_format_sources(self, sample_groups):
        from imas_codex.ids.mapping import _format_sources

        result = _format_sources(sample_groups)
        assert "pf_coils:group1" in result
        assert "domain=magnetic_field_systems" in result
        assert "members=6" in result

    def test_format_fields(self, sample_subtree):
        from imas_codex.ids.mapping import _format_fields

        result = _format_fields(sample_subtree[1:])
        assert "FLT_0D" in result
        assert "[m]" in result

    def test_format_empty(self):
        from imas_codex.ids.mapping import (
            _format_fields,
            _format_sources,
            _format_subtree,
        )

        assert _format_subtree([]) == "(no paths)"
        assert _format_sources([]) == "(no sources)"
        assert _format_fields([]) == "(no fields)"
