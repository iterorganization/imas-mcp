"""End-to-end tests for the IMAS mapping pipeline.

Tests the full pipeline chain: tools → models → orchestrator → CLI,
using mocked GraphClient and LLM calls to validate the integration
without requiring a live Neo4j database or LLM API.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.core.data_model import IdsNode
from imas_codex.ids.models import (
    EscalationFlag,
    EscalationSeverity,
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
    ValidatedMappingResult,
    ValidatedSignalMapping,
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
    search_imas_semantic,
)
from imas_codex.models.error_models import ToolError
from imas_codex.models.result_models import FetchPathsResult

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
    """Sample signal sources from a JET pf_active scenario."""
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
    return TargetAssignmentBatch(
        ids_name="pf_active",
        assignments=[
            TargetAssignment(
                source_id="jet:pf_coils:group1",
                imas_target_path="pf_active/coil",
                confidence=0.95,
                reasoning="PF coil geometry maps to pf_active/coil",
            ),
            TargetAssignment(
                source_id="jet:pf_coils:group2",
                imas_target_path="pf_active/coil",
                confidence=0.90,
                reasoning="PF coil 2 geometry maps to pf_active/coil",
            ),
        ],
        unassigned_groups=[],
    )


@pytest.fixture
def sample_field_batch():
    """Sample field mapping batch from Step 2."""
    return SignalMappingBatch(
        ids_name="pf_active",
        target_path="pf_active/coil",
        mappings=[
            SignalMappingEntry(
                source_id="jet:pf_coils:group1",
                source_property="value",
                target_id="pf_active/coil/element/geometry/rectangle/r",
                transform_expression="value",
                source_units="m",
                target_units="m",
                confidence=0.95,
                reasoning="Direct R position mapping",
            ),
            SignalMappingEntry(
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
            TargetAssignment(
                source_id="jet:pf_coils:group1",
                imas_target_path="pf_active/coil",
                confidence=0.95,
                reasoning="PF coil geometry maps to pf_active/coil",
            ),
        ],
        bindings=[
            ValidatedSignalMapping(
                source_id="jet:pf_coils:group1",
                source_property="value",
                target_id="pf_active/coil/element/geometry/rectangle/r",
                transform_expression="value",
                source_units="m",
                target_units="m",
                confidence=0.95,
            ),
            ValidatedSignalMapping(
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

    @patch("imas_codex.tools.graph_search.GraphPathTool.fetch_imas_paths", new_callable=AsyncMock)
    def test_returns_stable_dict_output_for_string_cluster_labels(
        self, mock_fetch_paths, mock_gc
    ):
        mock_fetch_paths.return_value = FetchPathsResult(
            nodes=[
                IdsNode(
                    id="pf_active/coil/current/data",
                    name="data",
                    ids="pf_active",
                    path="pf_active/coil/current/data",
                    documentation="Current data",
                    data_type="FLT_1D",
                    node_type="dynamic",
                    cluster_labels=["coil currents"],
                )
            ]
        )

        result = fetch_imas_fields(
            "pf_active",
            ["coil/current/data"],
            gc=mock_gc,
        )

        assert len(result) == 1
        assert result[0]["path"] == "pf_active/coil/current/data"
        assert result[0]["documentation"] == "Current data"
        assert result[0]["data_type"] == "FLT_1D"
        assert result[0]["node_type"] == "dynamic"
        assert result[0]["cluster_labels"] == ["coil currents"]


class TestSearchImasSemantic:
    @patch("imas_codex.tools.graph_search.GraphSearchTool.search_imas_paths", new_callable=AsyncMock)
    def test_degrades_cleanly_on_tool_error(self, mock_search, mock_gc):
        mock_search.return_value = ToolError(
            error="semantic search backend unavailable",
            suggestions=[],
        )

        result = search_imas_semantic("coil current", gc=mock_gc)

        assert result == []


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


class TestTargetAssignmentBatch:
    def test_serialization(self, sample_section_assignment):
        data = sample_section_assignment.model_dump()
        restored = TargetAssignmentBatch.model_validate(data)
        assert len(restored.assignments) == 2
        assert restored.ids_name == "pf_active"

    def test_json_roundtrip(self, sample_section_assignment):
        json_str = sample_section_assignment.model_dump_json()
        restored = TargetAssignmentBatch.model_validate_json(json_str)
        assert restored.assignments[0].confidence == 0.95


class TestSignalMappingBatch:
    def test_serialization(self, sample_field_batch):
        data = sample_field_batch.model_dump()
        restored = SignalMappingBatch.model_validate(data)
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
    def test_assign_targets(
        self, mock_call_llm, sample_groups, sample_subtree, sample_section_assignment
    ):
        """Test section assignment."""
        from imas_codex.ids.mapping import PipelineCost, assign_targets

        mock_call_llm.return_value = sample_section_assignment
        cost = PipelineCost()

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "semantic": sample_subtree,
        }

        result = assign_targets("jet", "pf_active", context, cost=cost)
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
    @patch("imas_codex.ids.mapping.assign_targets")
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
    @patch("imas_codex.ids.mapping.assign_targets")
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

        unassigned = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_target_path="pf_active/coil",
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
        assert "IMAS signal mapping pipeline" in result.output
        assert "magnetic_field_systems" in result.output
        assert "Union of filters" in result.output

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
        assert "signal mapping pipeline" in result.output.lower()

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.ids.tools.discover_mappable_ids")
    @patch("imas_codex.ids.mapping.generate_mapping")
    def test_map_run(self, mock_generate, mock_discover, mock_gc, sample_validated_result):
        from imas_codex.cli.map import map_cmd
        from imas_codex.ids.mapping import MappingResult, PipelineCost

        mock_discover.return_value = {
            "available_domains": ["magnetics"],
            "ids_targets": [{"ids_name": "pf_active", "domains": ["magnetics"], "source_count": 5}],
            "total_sources": 5,
        }
        mock_generate.return_value = MappingResult(
            mapping_id="jet:pf_active",
            validated=sample_validated_result,
            cost=PipelineCost(steps={"step1": 0.001, "step2": 0.002}),
            persisted=True,
            unassigned_groups=["jet:pf_coils:group3"],
            assembly=None,
        )

        runner = CliRunner()
        result = runner.invoke(
            map_cmd, ["jet", "-i", "pf_active", "--no-persist"]
        )
        assert result.exit_code == 0
        assert "jet:pf_active" in result.output
        assert "Bindings: 2" in result.output
        assert "Unassigned signal sources" in result.output
        assert "jet:pf_coils:group3" in result.output

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.ids.tools.discover_mappable_ids")
    @patch("imas_codex.ids.mapping.generate_mapping")
    def test_map_run_error(self, mock_generate, mock_discover, mock_gc):
        from imas_codex.cli.map import map_cmd

        mock_discover.return_value = {
            "available_domains": ["magnetics"],
            "ids_targets": [{"ids_name": "pf_active", "domains": ["magnetics"], "source_count": 5}],
            "total_sources": 5,
        }
        mock_generate.side_effect = ValueError("No signal sources found")

        runner = CliRunner()
        result = runner.invoke(map_cmd, ["jet", "-i", "pf_active"])
        # Error is caught gracefully in multi-IDS loop, summary shows no success
        assert "No IDS were successfully mapped" in result.output or result.exit_code == 0


class TestDiscoverMappableIds:
    def test_union_of_domain_and_ids_filters(self):
        from unittest.mock import MagicMock

        from imas_codex.ids.tools import discover_mappable_ids

        gc = MagicMock()
        gc.query.side_effect = [
            [
                {"domain": "equilibrium", "cnt": 20},
                {"domain": "magnetic_field_systems", "cnt": 132},
                {"domain": "plasma_control", "cnt": 64},
            ],
            [
                {
                    "ids_name": "equilibrium",
                    "domains": ["equilibrium"],
                },
                {
                    "ids_name": "magnetics",
                    "domains": ["equilibrium"],
                },
            ],
            [
                {
                    "ids_name": "pf_active",
                    "domains": [
                        "magnetic_field_systems",
                        "plasma_control",
                    ],
                },
            ],
        ]

        result = discover_mappable_ids(
            "jet",
            gc=gc,
            domains=["equilibrium"],
            ids_filter=["pf_active"],
        )

        assert result["available_domains"] == [
            "equilibrium",
            "magnetic_field_systems",
            "plasma_control",
        ]
        assert [row["ids_name"] for row in result["ids_targets"]] == [
            "equilibrium",
            "magnetics",
            "pf_active",
        ]
        assert result["ids_targets"][-1]["domains"] == [
            "magnetic_field_systems",
            "plasma_control",
        ]
        assert result["total_sources"] == 216


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    """Test that prompt templates exist and render correctly."""

    def test_target_assignment_prompt_exists(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        path = PROMPTS_DIR / "mapping" / "target_assignment.md"
        assert path.exists()
        prompt = path.read_text().lower()
        assert "signal sources" in prompt

    def test_signal_mapping_prompt_exists(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        path = PROMPTS_DIR / "mapping" / "signal_mapping.md"
        assert path.exists()
        prompt = path.read_text().lower()
        assert "signal" in prompt
        assert "transform" in prompt

    def test_validation_prompt_exists(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        path = PROMPTS_DIR / "mapping" / "validation.md"
        assert path.exists()
        prompt = path.read_text().lower()
        assert "valid" in prompt

    def test_target_assignment_prompt_renders(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "target_assignment",
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
            target_path="pf_active/coil",
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
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        # Transform examples are now in the system prompt
        prompt = (PROMPTS_DIR / "mapping" / "signal_mapping_system.md").read_text()
        assert "value * 1e-3" in prompt
        assert "math.radians(value)" in prompt
        assert "convert_units(value" in prompt
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


# ---------------------------------------------------------------------------
# Phase 9.1: Pipeline unit tests
# ---------------------------------------------------------------------------


class TestGatherContextFields:
    """Test that gather_context returns all enriched fields."""

    @patch("imas_codex.ids.mapping.search_imas_semantic")
    def test_context_has_enriched_source_fields(
        self, mock_semantic, mock_gc, sample_groups, sample_subtree
    ):
        """All expected enriched keys present in context signal sources."""
        from imas_codex.ids.mapping import gather_context

        mock_semantic.return_value = sample_subtree

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

        # Check enriched fields on signal sources
        for grp in ctx["groups"]:
            assert "rep_unit" in grp
            assert "rep_cocos" in grp
            assert "physics_domain" in grp
            assert "sample_accessors" in grp


class TestAssignSectionsValidOutput:
    """Test that section assignment produces valid IDS paths."""

    @patch("imas_codex.ids.mapping._call_llm")
    def test_all_paths_start_with_ids_name(
        self, mock_call_llm, sample_groups, sample_subtree, sample_section_assignment
    ):
        from imas_codex.ids.mapping import PipelineCost, assign_targets

        mock_call_llm.return_value = sample_section_assignment
        cost = PipelineCost()

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "semantic": sample_subtree,
        }

        result = assign_targets("jet", "pf_active", context, cost=cost)
        assert isinstance(result, TargetAssignmentBatch)
        for a in result.assignments:
            assert a.imas_target_path.startswith("pf_active/")


class TestMapSignalsMultiTarget:
    """Test that map_signals supports multi-target (same source → multiple targets)."""

    @patch("imas_codex.ids.mapping.fetch_source_code_refs")
    @patch("imas_codex.ids.mapping._call_llm")
    @patch("imas_codex.ids.mapping.fetch_imas_fields")
    @patch("imas_codex.ids.mapping.fetch_imas_subtree")
    def test_same_source_multiple_targets(
        self,
        mock_subtree,
        mock_fields,
        mock_call_llm,
        mock_code_refs,
        mock_gc,
        sample_groups,
        sample_subtree,
    ):
        from imas_codex.ids.mapping import PipelineCost, map_signals

        # LLM returns batch with same source mapped to two targets
        multi_target_batch = SignalMappingBatch(
            ids_name="pf_active",
            target_path="pf_active/coil",
            mappings=[
                SignalMappingEntry(
                    source_id="jet:pf_coils:group1",
                    target_id="pf_active/coil/element/geometry/rectangle/r",
                    transform_expression="value",
                    confidence=0.95,
                    reasoning="R position",
                ),
                SignalMappingEntry(
                    source_id="jet:pf_coils:group1",
                    target_id="pf_active/coil/element/geometry/rectangle/z",
                    transform_expression="value",
                    confidence=0.90,
                    reasoning="Z position",
                ),
            ],
        )

        mock_fields.return_value = sample_subtree[1:]
        mock_subtree.return_value = sample_subtree[1:]
        mock_call_llm.return_value = multi_target_batch
        mock_code_refs.return_value = []
        cost = PipelineCost()

        assignment = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_target_path="pf_active/coil",
                    confidence=0.95,
                    reasoning="PF coil geometry",
                ),
            ],
        )

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "cocos_paths": [],
            "existing": {"mapping": None, "sections": [], "bindings": []},
        }

        result = map_signals(
            "jet", "pf_active", assignment, context, gc=mock_gc, cost=cost
        )

        # Same source mapped to both targets
        source_ids = {m.source_id for m in result[0].mappings}
        target_ids = {m.target_id for m in result[0].mappings}
        assert len(source_ids) == 1  # same source
        assert len(target_ids) == 2  # different targets


class TestDiscoverAssemblyPatterns:
    """Test that discover_assembly returns valid AssemblyBatch."""

    @patch("imas_codex.ids.mapping.fetch_imas_subtree")
    @patch("imas_codex.ids.mapping._call_llm")
    def test_returns_assembly_batch(
        self,
        mock_call_llm,
        mock_subtree,
        mock_gc,
        sample_groups,
        sample_subtree,
        sample_section_assignment,
        sample_field_batch,
    ):
        from imas_codex.ids.mapping import PipelineCost, discover_assembly
        from imas_codex.ids.models import AssemblyConfig, AssemblyPattern

        mock_subtree.return_value = sample_subtree
        mock_call_llm.return_value = AssemblyConfig(
            target_path="pf_active/coil",
            pattern=AssemblyPattern.ARRAY_PER_NODE,
            confidence=0.85,
            reasoning="Each PF coil maps to one struct-array element",
        )
        cost = PipelineCost()

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
        }

        result = discover_assembly(
            "jet",
            "pf_active",
            sample_section_assignment,
            [sample_field_batch, sample_field_batch],
            context,
            gc=mock_gc,
            cost=cost,
        )

        assert result.ids_name == "pf_active"
        assert len(result.configs) == 2
        for cfg in result.configs:
            assert isinstance(cfg.pattern, AssemblyPattern)


class TestValidateMappingsCatchesUnitMismatch:
    """Test that validation catches identity transform with unit mismatch."""

    @patch("imas_codex.ids.validation.check_imas_paths")
    def test_identity_transform_unit_mismatch(self, mock_check, mock_gc):
        from imas_codex.ids.mapping import validate_mappings

        mock_check.return_value = [
            {"path": "pf_active/coil/element/geometry/rectangle/r", "exists": True}
        ]
        mock_gc.query.return_value = [{"id": "jet:pf_coils:group1"}]

        sections = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_target_path="pf_active/coil",
                    confidence=0.9,
                    reasoning="test",
                ),
            ],
        )
        field_batch = SignalMappingBatch(
            ids_name="pf_active",
            target_path="pf_active/coil",
            mappings=[
                SignalMappingEntry(
                    source_id="jet:pf_coils:group1",
                    target_id="pf_active/coil/element/geometry/rectangle/r",
                    transform_expression="value",
                    source_units="mm",
                    target_units="m",
                    confidence=0.9,
                    reasoning="R position",
                ),
            ],
        )

        with patch("imas_codex.ids.validation.analyze_units") as mock_units:
            mock_units.return_value = {"compatible": True, "conversion_factor": 0.001}
            result = validate_mappings(
                "jet", "pf_active", "4.1.1", sections, [field_batch], gc=mock_gc
            )

        # Should have escalation about identity transform with different units
        unit_escalations = [
            e for e in result.escalations if "units differ" in e.reason.lower()
        ]
        assert len(unit_escalations) == 1


class TestValidateMappingsCatchesCOCOSMissing:
    """Test that validation catches sign-flip path without COCOS handling."""

    @patch("imas_codex.ids.validation.check_imas_paths")
    def test_sign_flip_path_no_cocos(self, mock_check, mock_gc):
        from imas_codex.ids.mapping import validate_mappings

        target = "equilibrium/time_slice/profiles_1d/psi"
        mock_check.return_value = [{"path": target, "exists": True}]
        mock_gc.query.return_value = [{"id": "jet:eq:psi_group"}]

        sections = TargetAssignmentBatch(
            ids_name="equilibrium",
            assignments=[
                TargetAssignment(
                    source_id="jet:eq:psi_group",
                    imas_target_path="equilibrium/time_slice",
                    confidence=0.9,
                    reasoning="test",
                ),
            ],
        )
        field_batch = SignalMappingBatch(
            ids_name="equilibrium",
            target_path="equilibrium/time_slice",
            mappings=[
                SignalMappingEntry(
                    source_id="jet:eq:psi_group",
                    target_id=target,
                    transform_expression="value",
                    confidence=0.85,
                    reasoning="PSI profile",
                ),
            ],
        )

        # Patch get_sign_flip_paths to return our target
        with patch("imas_codex.ids.tools.get_sign_flip_paths") as mock_flip:
            mock_flip.return_value = [target]
            result = validate_mappings(
                "jet",
                "equilibrium",
                "4.1.1",
                sections,
                [field_batch],
                gc=mock_gc,
            )

        # Should have escalation about COCOS
        cocos_escalations = [
            e for e in result.escalations if "cocos" in e.reason.lower()
        ]
        assert len(cocos_escalations) == 1


class TestValidateSameSourceDiffTargetsOk:
    """Test one source → multiple targets produces no warning."""

    @patch("imas_codex.ids.validation.check_imas_paths")
    def test_no_warning_for_multi_target(self, mock_check, mock_gc):
        from imas_codex.ids.validation import validate_mapping

        mock_check.return_value = [
            {"path": "pf_active/coil/element/geometry/rectangle/r", "exists": True},
            {"path": "pf_active/coil/element/geometry/rectangle/z", "exists": True},
        ]
        mock_gc.query.return_value = [{"id": "jet:pf_coils:group1"}]

        b1 = ValidatedSignalMapping(
            source_id="jet:pf_coils:group1",
            target_id="pf_active/coil/element/geometry/rectangle/r",
            transform_expression="value",
            confidence=0.9,
        )
        b2 = ValidatedSignalMapping(
            source_id="jet:pf_coils:group1",
            target_id="pf_active/coil/element/geometry/rectangle/z",
            transform_expression="value",
            confidence=0.9,
        )

        report = validate_mapping([b1, b2], gc=mock_gc)
        assert report.all_passed is True
        assert report.duplicate_targets == []
        # No warnings about same source → different targets
        multi_src_warns = [
            e for e in report.escalations if "Multiple sources" in e.reason
        ]
        assert len(multi_src_warns) == 0


# ---------------------------------------------------------------------------
# Phase 9.2: Integration tests (marked, require Neo4j)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationMappingPipeline:
    """Integration tests requiring Neo4j."""

    @pytest.fixture(autouse=True)
    def _setup_graph_nodes(self):
        """Create prerequisite nodes for integration tests, clean up after."""
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        # Create SignalSource, IMASNode, and Facility nodes used by fixtures
        gc.query("MERGE (:Facility {id: 'jet'})")
        gc.query("MERGE (:SignalSource {id: 'jet:pf_coils:group1'})")
        gc.query("MERGE (:SignalSource {id: 'jet:pf_coils:group2'})")
        gc.query(
            "MERGE (:IMASNode {id: 'pf_active/coil/element/geometry/rectangle/r'})"
        )
        gc.query(
            "MERGE (:IMASNode {id: 'pf_active/coil/element/geometry/rectangle/z'})"
        )
        gc.query("MERGE (:IMASNode {id: 'pf_active/coil'})")
        yield
        # Clean up mapping artefacts
        gc.query(
            """
            MATCH (m:IMASMapping {facility_id: 'jet', ids_name: 'pf_active'})
            OPTIONAL MATCH (m)-[:USES_SIGNAL_SOURCE]->(sg)
            OPTIONAL MATCH (sg)-[r:MAPS_TO_IMAS]->()
            OPTIONAL MATCH (sg)-[:HAS_EVIDENCE]->(ev)
            DETACH DELETE ev, r, m
            """
        )

    def test_persist_and_load_roundtrip(self, sample_validated_result):
        """Persist result → load → verify bindings match."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.ids.tools import search_existing_mappings

        gc = GraphClient()
        persist_mapping_result(sample_validated_result, gc=gc)
        loaded = search_existing_mappings("jet", "pf_active", gc=gc)
        assert loaded["mapping"] is not None
        assert len(loaded["bindings"]) == len(sample_validated_result.bindings)

    def test_assembly_config_persisted_on_populates(
        self, sample_validated_result
    ):
        """Assembly config properties appear on POPULATES relationships."""
        from imas_codex.graph.client import GraphClient
        from imas_codex.ids.models import (
            AssemblyBatch,
            AssemblyConfig,
            AssemblyPattern,
        )

        gc = GraphClient()
        assembly = AssemblyBatch(
            ids_name="pf_active",
            configs=[
                AssemblyConfig(
                    target_path="pf_active/coil",
                    pattern=AssemblyPattern.ARRAY_PER_NODE,
                    init_arrays={"coil": 12},
                    reasoning="One element per coil",
                    confidence=0.9,
                ),
            ],
        )
        persist_mapping_result(
            sample_validated_result, assembly=assembly, gc=gc
        )
        rows = gc.query(
            """
            MATCH (m:IMASMapping {facility_id: 'jet', ids_name: 'pf_active'})
                  -[r:POPULATES]->(s:IMASNode)
            RETURN r.structure AS pattern, r.init_arrays AS init_arrays
            """
        )
        assert len(rows) > 0
        assert rows[0]["pattern"] == "array_per_node"


# ---------------------------------------------------------------------------
# Phase 9.3: Prompt template tests
# ---------------------------------------------------------------------------


class TestAssemblyPromptRenders:
    def test_assembly_prompt_renders_all_vars(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "assembly",
            facility="jet",
            ids_name="pf_active",
            target_path="pf_active/coil",
            signal_mappings="- group1 → r (FLT_0D)",
            imas_section_structure="pf_active/coil (STRUCT_ARRAY)",
            source_metadata="2 groups, 6 signals",
        )
        assert "pf_active/coil" in rendered
        assert "jet" in rendered


class TestPromptsNoSignalMappingTerminology:
    def test_no_field_mapping_in_prompts(self):
        """No standalone 'field mapping' terminology in prompt templates.

        Occurrences as part of class names (ValidatedSignalMapping,
        SignalMappingEntry) are acceptable.
        """
        import os
        import re

        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        prompts_dir = PROMPTS_DIR / "mapping"
        # Match "field mapping" but not when preceded by "Validated" or
        # followed by "Entry", "Batch" (Pydantic class name contexts)
        pattern = re.compile(
            r"(?<!validated)field\s+mapping(?!entry|batch|s?\s*\()",
            re.IGNORECASE,
        )
        for fname in os.listdir(prompts_dir):
            if fname.endswith(".md"):
                content = (prompts_dir / fname).read_text()
                matches = pattern.findall(content)
                assert not matches, (
                    f"Found standalone 'field mapping' in {fname}: {matches}"
                )


class TestStaticFirstOrdering:
    def test_static_in_system_prompt_dynamic_in_user_prompt(self):
        """Static instructions are in the system prompt file, dynamic in user."""
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        system = (PROMPTS_DIR / "mapping" / "signal_mapping_system.md").read_text()
        user = (PROMPTS_DIR / "mapping" / "signal_mapping.md").read_text()

        # Static instructions live in system prompt
        assert "Task" in system
        assert "Transform Rules" in system
        assert "Output Format" in system

        # Dynamic context lives in user prompt
        assert "{{ facility }}" in user
        assert "{{ ids_name }}" in user

        # System prompt has NO template variables
        assert "{{ " not in system

    def test_all_system_prompts_are_static(self):
        """Verify none of the system prompt files contain Jinja2 variables."""
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        for name in [
            "signal_mapping_system.md",
            "target_assignment_system.md",
            "assembly_system.md",
        ]:
            content = (PROMPTS_DIR / "mapping" / name).read_text()
            assert "{{ " not in content, f"{name} should not contain template variables"


# ---------------------------------------------------------------------------
# Phase 9.4: CLI tests
# ---------------------------------------------------------------------------


class TestMapRunCostLimit:
    @patch("imas_codex.ids.tools.discover_mappable_ids")
    @patch("imas_codex.ids.mapping.generate_mapping")
    def test_cost_limit_flag_accepted(self, mock_generate, mock_discover, sample_validated_result):
        from imas_codex.cli.map import map_cmd
        from imas_codex.ids.mapping import MappingResult, PipelineCost

        mock_discover.return_value = {
            "available_domains": ["magnetics"],
            "ids_targets": [{"ids_name": "pf_active", "domains": ["magnetics"], "source_count": 5}],
            "total_sources": 5,
        }
        mock_generate.return_value = MappingResult(
            mapping_id="jet:pf_active",
            validated=sample_validated_result,
            cost=PipelineCost(steps={"step1": 0.001}),
            persisted=False,
            unassigned_groups=[],
            assembly=None,
        )

        runner = CliRunner()
        result = runner.invoke(
            map_cmd,
            ["jet", "-i", "pf_active", "--cost-limit", "2.0", "--no-persist"],
        )
        assert result.exit_code == 0
        assert "jet:pf_active" in result.output


class TestMapRunTimeLimit:
    @patch("imas_codex.ids.tools.discover_mappable_ids")
    @patch("imas_codex.ids.mapping.generate_mapping")
    def test_time_flag_accepted(self, mock_generate, mock_discover, sample_validated_result):
        from imas_codex.cli.map import map_cmd
        from imas_codex.ids.mapping import MappingResult, PipelineCost

        mock_discover.return_value = {
            "available_domains": ["magnetics"],
            "ids_targets": [{"ids_name": "pf_active", "domains": ["magnetics"], "source_count": 5}],
            "total_sources": 5,
        }
        mock_generate.return_value = MappingResult(
            mapping_id="jet:pf_active",
            validated=sample_validated_result,
            cost=PipelineCost(steps={"step1": 0.001}),
            persisted=False,
            unassigned_groups=[],
            assembly=None,
        )

        runner = CliRunner()
        result = runner.invoke(
            map_cmd,
            ["jet", "-i", "pf_active", "--time", "10", "--no-persist"],
        )
        assert result.exit_code == 0
        assert "jet:pf_active" in result.output


# ---------------------------------------------------------------------------
# Phase 8: E2E tests for pipeline plan changes
# ---------------------------------------------------------------------------


class TestTransformExpressionValidator:
    """Test the transform_expression validator on SignalMappingEntry."""

    def test_identity_transform(self):
        e = SignalMappingEntry(
            source_id="s1", target_id="t1",
            transform_expression="value", confidence=0.9,
        )
        assert e.transform_expression == "value"

    def test_empty_transform_defaults(self):
        e = SignalMappingEntry(
            source_id="s1", target_id="t1", confidence=0.9,
        )
        assert e.transform_expression == "value"

    def test_arithmetic_transforms(self):
        for expr in ["-value", "value * 1e-3", "value + 1.0", "value / 2"]:
            e = SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression=expr, confidence=0.9,
            )
            assert e.transform_expression == expr

    def test_allowed_function_calls(self):
        for expr in [
            "convert_units(value, 'mm', 'm')",
            "math.radians(value)",
            "abs(value)",
            "cocos_sign('ip_like', cocos_in=2, cocos_out=11) * value",
        ]:
            e = SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression=expr, confidence=0.9,
            )
            assert e.transform_expression == expr

    def test_blocks_import(self):
        with pytest.raises(ValueError):
            SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression="import os", confidence=0.5,
            )

    def test_blocks_eval(self):
        with pytest.raises(ValueError):
            SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression="eval('1+1')", confidence=0.5,
            )

    def test_blocks_exec(self):
        with pytest.raises(ValueError):
            SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression="exec('x=1')", confidence=0.5,
            )

    def test_blocks_dunder(self):
        with pytest.raises(ValueError):
            SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression="__builtins__", confidence=0.5,
            )

    def test_blocks_getattr(self):
        with pytest.raises(ValueError):
            SignalMappingEntry(
                source_id="s1", target_id="t1",
                transform_expression="getattr(value, '__class__')",
                confidence=0.5,
            )


class TestCoverageThreshold:
    """Test the coverage threshold enforcement."""

    def test_import(self):
        from imas_codex.ids.validation import (
            COVERAGE_ERROR_THRESHOLD,
            COVERAGE_WARNING_THRESHOLD,
            check_coverage_threshold,
        )
        assert COVERAGE_WARNING_THRESHOLD > COVERAGE_ERROR_THRESHOLD
        assert COVERAGE_ERROR_THRESHOLD >= 0

    def test_check_returns_list(self):
        from imas_codex.ids.validation import check_coverage_threshold

        # With mock gc that returns no fields, coverage = 0%
        gc = MagicMock()
        gc.query.return_value = []
        result = check_coverage_threshold("pf_active", [], gc=gc)
        assert isinstance(result, list)


class TestMappingDiscoveryState:
    """Test the MappingDiscoveryState from workers.py."""

    def test_create(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(
            facility="jet", target_ids="pf_active", cost_limit=5.0,
        )
        assert state.facility == "jet"
        assert state.target_ids == "pf_active"
        assert state.cost_limit == 5.0
        assert not state.should_stop()
        assert state.total_cost == 0.0

    def test_phases_initially_not_done(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(facility="jet", target_ids="pf_active")
        assert not state.context_phase.done
        assert not state.assign_phase.done
        assert not state.map_phase.done
        assert not state.validate_phase.done

    def test_phase_mark_done(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(facility="jet", target_ids="pf_active")
        state.context_phase.mark_done()
        assert state.context_phase.done
        assert not state.assign_phase.done

    def test_should_stop_when_requested(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(facility="jet", target_ids="pf_active")
        assert not state.should_stop()
        state.stop_requested = True
        assert state.should_stop()

    def test_should_stop_when_budget_exhausted(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(
            facility="jet", target_ids="pf_active", cost_limit=1.0,
        )
        state.cost.add("test", 2.0, 1000)
        assert state.should_stop()

    def test_mapping_id_format(self):
        """Mapping ID is facility:ids_name (DD version stored as property)."""
        result = ValidatedMappingResult(
            facility="jet", ids_name="pf_active", dd_version="4.1.1",
            sections=[], bindings=[], escalations=[],
        )

        gc = MagicMock()
        gc.query.return_value = []
        mapping_id = persist_mapping_result(result, gc=gc)
        assert mapping_id == "jet:pf_active"


class TestWorkerImports:
    """Test that all worker components import correctly."""

    def test_worker_functions_importable(self):
        from imas_codex.ids.workers import (
            assign_worker,
            claim_sources_for_mapping,
            context_worker,
            map_worker,
            release_source_claim,
            run_mapping_engine,
            validate_worker,
        )
        assert callable(context_worker)
        assert callable(assign_worker)
        assert callable(map_worker)
        assert callable(validate_worker)
        assert callable(run_mapping_engine)
        assert callable(claim_sources_for_mapping)
        assert callable(release_source_claim)


class TestSemanticCandidatesInPrompt:
    """Test that semantic_candidates renders in signal_mapping prompt."""

    def test_semantic_candidates_placeholder(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "signal_mapping",
            facility="jet",
            ids_name="pf_active",
            target_path="pf_active/coil",
            signal_source_detail="test",
            imas_fields="test",
            identifier_schemas="",
            version_context="",
            unit_analysis="",
            cocos_paths="(none)",
            existing_mappings="{}",
            code_references="(none)",
            source_cocos="(none)",
            semantic_candidates="- pf_active/coil/r (score=0.95): Coil R position",
        )
        assert "Semantic Candidates" in rendered
        assert "pf_active/coil/r" in rendered
        assert "score=0.95" in rendered
