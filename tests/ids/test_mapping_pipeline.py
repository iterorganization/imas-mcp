"""Tests for the mapping pipeline cache optimization and prompt architecture.

Validates that:
1. System prompts are static and contain no template variables
2. _build_messages constructs correct system/user message pairs
3. System prompts are identical across calls (enabling LLM cache hits)
4. Prompt templates render correctly with realistic context
5. Pipeline stages produce structurally valid outputs
6. Wiki/code/semantic formatters handle edge cases
7. Validation helpers (classify_many_to_one, coverage threshold) work correctly
8. MappingDisposition and UnmappedSignal serialize correctly
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.ids.models import (
    AssemblyConfig,
    AssemblyPattern,
    EscalationFlag,
    EscalationSeverity,
    MappingDisposition,
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
    UnassignedSource,
    UnmappedSignal,
    ValidatedSignalMapping,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gc():
    gc = MagicMock()
    gc.query.return_value = []
    return gc


@pytest.fixture
def sample_groups():
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
    ]


@pytest.fixture
def sample_subtree():
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


# ---------------------------------------------------------------------------
# 1. Cache optimization: system/user prompt split
# ---------------------------------------------------------------------------


class TestSystemPromptSplit:
    """Verify prompt split: static system + dynamic user."""

    def test_system_prompts_have_no_template_vars(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        for name in [
            "signal_mapping_system.md",
            "target_assignment_system.md",
            "assembly_system.md",
        ]:
            content = (PROMPTS_DIR / "mapping" / name).read_text()
            assert "{{ " not in content, f"{name} contains Jinja2 variables"

    def test_user_prompts_have_template_vars(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        for name in [
            "signal_mapping.md",
            "target_assignment.md",
            "assembly.md",
        ]:
            content = (PROMPTS_DIR / "mapping" / name).read_text()
            assert "{{ " in content, f"{name} should contain template variables"

    def test_system_prompts_contain_instructions(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        sig = (PROMPTS_DIR / "mapping" / "signal_mapping_system.md").read_text()
        assert "Task" in sig
        assert "Transform Rules" in sig
        assert "Output Format" in sig
        assert "No-Match Handling" in sig
        assert "Many-to-One Mappings" in sig

        sec = (PROMPTS_DIR / "mapping" / "target_assignment_system.md").read_text()
        assert "Task" in sec
        assert "Output Format" in sec
        assert "UnassignedSource" in sec

        asm = (PROMPTS_DIR / "mapping" / "assembly_system.md").read_text()
        assert "Assembly Patterns" in asm
        assert "Code Generation" in asm
        assert "array_per_node" in asm

    def test_user_prompts_have_context_sections(self):
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        sig = (PROMPTS_DIR / "mapping" / "signal_mapping.md").read_text()
        assert "{{ facility }}" in sig
        assert "{{ imas_fields }}" in sig
        assert "{{ cocos_paths }}" in sig

        sec = (PROMPTS_DIR / "mapping" / "target_assignment.md").read_text()
        assert "{{ facility }}" in sec
        assert "{{ signal_sources }}" in sec
        assert "{{ imas_subtree }}" in sec

        asm = (PROMPTS_DIR / "mapping" / "assembly.md").read_text()
        assert "{{ facility }}" in asm
        assert "{{ signal_mappings }}" in asm


# ---------------------------------------------------------------------------
# 2. _build_messages construction
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Verify _build_messages creates correct message pairs."""

    def test_returns_system_and_user(self):
        from imas_codex.ids.mapping import _build_messages

        messages = _build_messages(
            "signal_mapping_system",
            "User context goes here",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User context goes here"

    def test_system_content_is_from_template(self):
        from imas_codex.ids.mapping import _build_messages

        messages = _build_messages("signal_mapping_system", "user prompt")
        system_content = messages[0]["content"]
        assert "IMAS mapping expert" in system_content
        assert "Transform Rules" in system_content

    def test_system_prompt_identical_across_calls(self):
        """Same system template renders identically — enables cache hits."""
        from imas_codex.ids.mapping import _build_messages

        m1 = _build_messages("signal_mapping_system", "context A")
        m2 = _build_messages("signal_mapping_system", "context B")

        assert m1[0]["content"] == m2[0]["content"]  # identical system
        assert m1[1]["content"] != m2[1]["content"]  # different user

    def test_different_system_templates_differ(self):
        from imas_codex.ids.mapping import _build_messages

        m1 = _build_messages("signal_mapping_system", "x")
        m2 = _build_messages("target_assignment_system", "x")

        assert m1[0]["content"] != m2[0]["content"]

    def test_target_assignment_system_prompt(self):
        from imas_codex.ids.mapping import _build_messages

        messages = _build_messages("target_assignment_system", "user prompt")
        assert "IMAS mapping expert" in messages[0]["content"]
        assert "TargetAssignmentBatch" in messages[0]["content"]

    def test_assembly_system_prompt(self):
        from imas_codex.ids.mapping import _build_messages

        messages = _build_messages("assembly_system", "user prompt")
        assert "assembly expert" in messages[0]["content"]
        assert "array_per_node" in messages[0]["content"]


# ---------------------------------------------------------------------------
# 3. Prompt rendering with realistic context
# ---------------------------------------------------------------------------


class TestPromptRenderingWithContext:
    """Test that prompts render correctly with realistic data."""

    def test_target_assignment_renders_all_sections(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "target_assignment",
            facility="jet",
            ids_name="pf_active",
            signal_sources="- jet:pf_coils:group1 (domain=magnetic_field_systems)",
            imas_subtree="pf_active/coil (STRUCT_ARRAY) — PF coil descriptions",
            semantic_results="pf_active/coil (0.95): PF coil details",
            section_clusters="- **Magnetic Systems**: pf_active/coil",
            cross_facility_mappings="- **tcv**: pf_active/coil",
        )
        assert "jet" in rendered
        assert "pf_active" in rendered
        assert "jet:pf_coils:group1" in rendered
        assert "tcv" in rendered

    def test_target_assignment_omits_cross_facility_when_empty(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "target_assignment",
            facility="jet",
            ids_name="pf_active",
            signal_sources="- src1",
            imas_subtree="tree",
            semantic_results="results",
            section_clusters="clusters",
            cross_facility_mappings="",
        )
        assert "Cross-Facility" not in rendered

    def test_signal_mapping_renders_with_all_context(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "signal_mapping",
            facility="jet",
            ids_name="pf_active",
            target_path="pf_active/coil",
            signal_source_detail="Source: jet:pf_coils:group1",
            imas_fields="- pf_active/coil/element/geometry/rectangle/r (FLT_0D) [m]",
            identifier_schemas="(no identifier schemas)",
            unit_analysis="m → m: compatible",
            cocos_paths="(none)",
            existing_mappings="{}",
            code_references="(no code references available)",
            source_cocos="(no COCOS context)",
            semantic_candidates="pf_active/coil/element (0.90): Geometry",
            cluster_candidates="(no cluster candidates)",
            version_context="(no version change history)",
            wiki_context="**PF Active** (IMAS relevance: 0.85)\nThe pf_active IDS...",
            code_data_access="scripts/read_jet.py::read_pf (data_access: 0.90)\n```python\ndef read_pf():\n    pass\n```",
            semantic_match_matrix="- IMAS: pf_active/coil/element (0.92)",
        )
        assert "jet" in rendered
        assert "pf_active/coil" in rendered
        assert "Domain Documentation" in rendered
        assert "PF Active" in rendered
        assert "Data Access Code" in rendered
        assert "Semantic Match Matrix" in rendered
        assert "read_jet.py" in rendered

    def test_signal_mapping_omits_optional_sections_when_empty(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "signal_mapping",
            facility="jet",
            ids_name="pf_active",
            target_path="pf_active/coil",
            signal_source_detail="source detail",
            imas_fields="fields",
            identifier_schemas="schemas",
            unit_analysis="units",
            cocos_paths="(none)",
            existing_mappings="{}",
            code_references="(none)",
            source_cocos="(none)",
            semantic_candidates="(none)",
            cluster_candidates="(none)",
            version_context="(none)",
            wiki_context="",
            code_data_access="",
            semantic_match_matrix="",
        )
        # Conditional sections should not appear
        assert "Domain Documentation" not in rendered
        assert "Data Access Code" not in rendered
        assert "Semantic Match Matrix" not in rendered

    def test_assembly_renders_with_context(self):
        from imas_codex.ids.mapping import _render_prompt

        rendered = _render_prompt(
            "assembly",
            facility="jet",
            ids_name="pf_active",
            target_path="pf_active/coil",
            signal_mappings="- jet:pf_coils:group1.value → pf_active/coil/element/geometry/rectangle/r",
            imas_section_structure="pf_active/coil/element (STRUCT_ARRAY)",
            source_metadata="Source: jet:pf_coils:group1",
            identifier_schemas="(no identifier schemas)",
            coordinate_context="(no coordinate spec data)",
        )
        assert "jet" in rendered
        assert "pf_active/coil" in rendered
        assert "group1" in rendered


# ---------------------------------------------------------------------------
# 4. Wiki/code/semantic match formatters
# ---------------------------------------------------------------------------


class TestFormatWikiContext:
    def test_empty(self):
        from imas_codex.ids.mapping import _format_wiki_context

        assert _format_wiki_context([]) == ""

    def test_single_item(self):
        from imas_codex.ids.mapping import _format_wiki_context

        items = [
            {
                "page_title": "PF Active",
                "text": "The pf_active IDS describes active PF coils.",
                "score_imas_relevance": 0.85,
            }
        ]
        result = _format_wiki_context(items)
        assert "**PF Active**" in result
        assert "0.85" in result
        assert "pf_active IDS" in result

    def test_long_text_truncated(self):
        from imas_codex.ids.mapping import _format_wiki_context

        items = [
            {
                "page_title": "Long",
                "text": "x" * 600,
                "score_imas_relevance": 0.5,
            }
        ]
        result = _format_wiki_context(items)
        assert "..." in result
        # Truncated to 500 chars + "..."
        assert "x" * 500 + "..." in result

    def test_no_title(self):
        from imas_codex.ids.mapping import _format_wiki_context

        items = [{"text": "just text", "score_imas_relevance": 0.3}]
        result = _format_wiki_context(items)
        assert "just text" in result


class TestFormatCodeContext:
    def test_empty(self):
        from imas_codex.ids.mapping import _format_code_context

        assert _format_code_context([]) == ""

    def test_single_item(self):
        from imas_codex.ids.mapping import _format_code_context

        items = [
            {
                "source_file": "scripts/read_jet.py",
                "function_name": "read_pf",
                "text": "def read_pf():\n    pass",
                "language": "python",
                "score_data_access": 0.92,
            }
        ]
        result = _format_code_context(items)
        assert "read_jet.py::read_pf" in result
        assert "0.92" in result
        assert "```python" in result

    def test_no_function_name(self):
        from imas_codex.ids.mapping import _format_code_context

        items = [
            {
                "source_file": "lib/utils.py",
                "text": "# utility code",
                "language": "python",
                "score_data_access": 0.5,
            }
        ]
        result = _format_code_context(items)
        assert "lib/utils.py" in result
        assert "::" not in result.split("\n")[0]

    def test_long_text_truncated(self):
        from imas_codex.ids.mapping import _format_code_context

        items = [
            {
                "source_file": "long.py",
                "text": "x" * 1000,
                "language": "python",
                "score_data_access": 0.3,
            }
        ]
        result = _format_code_context(items)
        assert "..." in result


class TestFormatSemanticMatchMatrix:
    def test_empty_matrix(self):
        from imas_codex.ids.mapping import _format_semantic_match_matrix

        assert _format_semantic_match_matrix({}, "src1") == ""

    def test_source_not_in_matrix(self):
        from imas_codex.ids.mapping import _format_semantic_match_matrix

        matrix = {
            "other_src": [{"content_type": "imas", "target_id": "a/b", "score": 0.9}]
        }
        assert _format_semantic_match_matrix(matrix, "src1") == ""

    def test_formats_matches_by_type(self):
        from imas_codex.ids.mapping import _format_semantic_match_matrix

        matrix = {
            "src1": [
                {
                    "content_type": "imas",
                    "target_id": "pf_active/coil/r",
                    "score": 0.92,
                    "excerpt": "R position",
                },
                {
                    "content_type": "wiki",
                    "target_id": "PF Active page",
                    "score": 0.85,
                    "excerpt": "",
                },
                {
                    "content_type": "code",
                    "target_id": "read_jet.py",
                    "score": 0.78,
                    "excerpt": "def read_pf",
                },
            ]
        }
        result = _format_semantic_match_matrix(matrix, "src1")
        assert "IMAS: pf_active/coil/r (0.920)" in result
        assert "Wiki: PF Active page (0.850)" in result
        assert "Code: read_jet.py (0.780)" in result
        assert "R position" in result


# ---------------------------------------------------------------------------
# 5. Model serialization
# ---------------------------------------------------------------------------


class TestMappingDisposition:
    def test_all_values(self):
        values = set(MappingDisposition)
        expected = {
            "mapped",
            "no_imas_equivalent",
            "metadata_only",
            "facility_specific",
            "insufficient_context",
        }
        assert {v.value for v in values} == expected

    def test_str_enum_serialization(self):
        assert str(MappingDisposition.NO_IMAS_EQUIVALENT) == "no_imas_equivalent"


class TestUnmappedSignal:
    def test_basic_serialization(self):
        u = UnmappedSignal(
            source_id="jet:pf:calibration",
            disposition=MappingDisposition.METADATA_ONLY,
            evidence="Calibration timestamps are not measured quantities",
        )
        data = u.model_dump()
        assert data["disposition"] == "metadata_only"
        assert data["nearest_imas_path"] is None

    def test_with_nearest_path(self):
        u = UnmappedSignal(
            source_id="jet:misc:special",
            disposition=MappingDisposition.NO_IMAS_EQUIVALENT,
            evidence="No IDS field for this measurement",
            nearest_imas_path="pf_active/coil/element/geometry/rectangle/r",
            nearest_similarity=0.45,
        )
        data = u.model_dump()
        assert (
            data["nearest_imas_path"] == "pf_active/coil/element/geometry/rectangle/r"
        )
        assert data["nearest_similarity"] == pytest.approx(0.45)

    def test_roundtrip(self):
        u = UnmappedSignal(
            source_id="src",
            disposition=MappingDisposition.INSUFFICIENT_CONTEXT,
            evidence="Could map but not enough evidence",
        )
        restored = UnmappedSignal.model_validate_json(u.model_dump_json())
        assert restored.disposition == MappingDisposition.INSUFFICIENT_CONTEXT
        assert restored.evidence == u.evidence


class TestUnassignedSource:
    def test_serialization(self):
        u = UnassignedSource(
            source_id="jet:misc:telemetry",
            disposition=MappingDisposition.FACILITY_SPECIFIC,
            evidence="JET-specific telemetry with no IDS coverage",
        )
        data = u.model_dump()
        assert data["disposition"] == "facility_specific"
        assert data["source_id"] == "jet:misc:telemetry"

    def test_roundtrip(self):
        u = UnassignedSource(
            source_id="src",
            disposition=MappingDisposition.INSUFFICIENT_CONTEXT,
            evidence="Might map but unclear",
        )
        restored = UnassignedSource.model_validate_json(u.model_dump_json())
        assert restored.disposition == MappingDisposition.INSUFFICIENT_CONTEXT


class TestSignalMappingBatchWithUnmapped:
    def test_batch_with_unmapped(self):
        batch = SignalMappingBatch(
            ids_name="pf_active",
            target_path="pf_active/coil",
            mappings=[
                SignalMappingEntry(
                    source_id="jet:pf:group1",
                    target_id="pf_active/coil/element/geometry/rectangle/r",
                    transform_expression="value",
                    confidence=0.95,
                    reasoning="R position",
                ),
            ],
            unmapped=[
                UnmappedSignal(
                    source_id="jet:pf:calibration",
                    disposition=MappingDisposition.METADATA_ONLY,
                    evidence="Calibration data, not a physical measurement",
                ),
            ],
        )
        data = batch.model_dump()
        assert len(data["unmapped"]) == 1
        assert data["unmapped"][0]["disposition"] == "metadata_only"

    def test_batch_json_roundtrip_with_unmapped(self):
        batch = SignalMappingBatch(
            ids_name="eq",
            target_path="equilibrium/time_slice",
            mappings=[],
            unmapped=[
                UnmappedSignal(
                    source_id="jet:eq:debug",
                    disposition=MappingDisposition.FACILITY_SPECIFIC,
                    evidence="Debug output",
                ),
            ],
        )
        restored = SignalMappingBatch.model_validate_json(batch.model_dump_json())
        assert len(restored.unmapped) == 1
        assert restored.unmapped[0].source_id == "jet:eq:debug"


class TestTargetAssignmentBatchWithUnassigned:
    def test_batch_with_unassigned(self):
        batch = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf:group1",
                    imas_target_path="pf_active/coil",
                    confidence=0.95,
                    reasoning="test",
                ),
            ],
            unassigned=[
                UnassignedSource(
                    source_id="jet:pf:telemetry",
                    disposition=MappingDisposition.FACILITY_SPECIFIC,
                    evidence="JET-only telemetry",
                ),
            ],
        )
        assert len(batch.unassigned) == 1
        assert batch.unassigned[0].disposition == MappingDisposition.FACILITY_SPECIFIC

    def test_batch_roundtrip_with_unassigned(self):
        batch = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[],
            unassigned=[
                UnassignedSource(
                    source_id="src",
                    disposition=MappingDisposition.NO_IMAS_EQUIVALENT,
                    evidence="No IDS section",
                ),
            ],
        )
        restored = TargetAssignmentBatch.model_validate_json(batch.model_dump_json())
        assert len(restored.unassigned) == 1


# ---------------------------------------------------------------------------
# 6. Validation helpers
# ---------------------------------------------------------------------------


class TestClassifyManyToOne:
    def test_single_source_returns_legitimate(self, mock_gc):
        from imas_codex.ids.validation import (
            DuplicateTargetClassification,
            _classify_many_to_one,
        )

        bindings = [MagicMock(source_id="src1")]
        result = _classify_many_to_one("target", bindings, mock_gc)
        assert result == DuplicateTargetClassification.LEGITIMATE_OTHER

    def test_same_prefix_returns_epoch_variants(self, mock_gc):
        from imas_codex.ids.validation import (
            DuplicateTargetClassification,
            _classify_many_to_one,
        )

        mock_gc.query.return_value = [
            {
                "id": "jet:pf:coil:1",
                "group_key": "pf_coil:1",
                "physics_domain": "magnetic",
                "description": "",
            },
            {
                "id": "jet:pf:coil:2",
                "group_key": "pf_coil:2",
                "physics_domain": "magnetic",
                "description": "",
            },
        ]
        bindings = [
            MagicMock(source_id="jet:pf:coil:1"),
            MagicMock(source_id="jet:pf:coil:2"),
        ]
        result = _classify_many_to_one("target", bindings, mock_gc)
        assert result == DuplicateTargetClassification.EPOCH_VARIANTS

    def test_same_domain_returns_processing_stages(self, mock_gc):
        from imas_codex.ids.validation import (
            DuplicateTargetClassification,
            _classify_many_to_one,
        )

        mock_gc.query.return_value = [
            {
                "id": "jet:eq:psi_raw",
                "group_key": "raw_psi",
                "physics_domain": "equilibrium",
                "description": "",
            },
            {
                "id": "jet:eq:psi_filtered",
                "group_key": "filtered_psi",
                "physics_domain": "equilibrium",
                "description": "",
            },
        ]
        bindings = [
            MagicMock(source_id="jet:eq:psi_raw"),
            MagicMock(source_id="jet:eq:psi_filtered"),
        ]
        result = _classify_many_to_one("target", bindings, mock_gc)
        assert result == DuplicateTargetClassification.PROCESSING_STAGES

    def test_different_domains_returns_erroneous(self, mock_gc):
        from imas_codex.ids.validation import (
            DuplicateTargetClassification,
            _classify_many_to_one,
        )

        mock_gc.query.return_value = [
            {
                "id": "src_a",
                "group_key": "a",
                "physics_domain": "magnetic",
                "description": "",
            },
            {
                "id": "src_b",
                "group_key": "b",
                "physics_domain": "thermal",
                "description": "",
            },
        ]
        bindings = [MagicMock(source_id="src_a"), MagicMock(source_id="src_b")]
        result = _classify_many_to_one("target", bindings, mock_gc)
        assert result == DuplicateTargetClassification.ERRONEOUS


class TestCheckCoverageThreshold:
    def test_good_coverage_no_escalations(self, mock_gc):
        from imas_codex.ids.validation import check_coverage_threshold

        with patch("imas_codex.ids.validation.compute_coverage") as mock_cov:
            mock_cov.return_value = MagicMock(
                mapped_fields=50, total_leaf_fields=100, percentage=50.0
            )
            result = check_coverage_threshold("pf_active", [], gc=mock_gc)
            assert result == []

    def test_low_coverage_warning(self, mock_gc):
        from imas_codex.ids.validation import check_coverage_threshold

        with patch("imas_codex.ids.validation.compute_coverage") as mock_cov:
            mock_cov.return_value = MagicMock(
                mapped_fields=3, total_leaf_fields=100, percentage=3.0
            )
            result = check_coverage_threshold("pf_active", [], gc=mock_gc)
            assert len(result) == 1
            assert result[0].severity == EscalationSeverity.WARNING

    def test_critical_low_coverage_error(self, mock_gc):
        from imas_codex.ids.validation import check_coverage_threshold

        with patch("imas_codex.ids.validation.compute_coverage") as mock_cov:
            mock_cov.return_value = MagicMock(
                mapped_fields=0, total_leaf_fields=200, percentage=0.0
            )
            result = check_coverage_threshold("pf_active", [], gc=mock_gc)
            assert len(result) == 1
            assert result[0].severity == EscalationSeverity.ERROR


# ---------------------------------------------------------------------------
# 7. Pipeline integration: assign_targets uses _build_messages
# ---------------------------------------------------------------------------


class TestAssignSectionsUsesSystemPrompt:
    """Verify assign_targets sends system+user messages."""

    @patch("imas_codex.ids.mapping._call_llm")
    def test_messages_have_system_role(
        self, mock_call_llm, sample_groups, sample_subtree
    ):
        from imas_codex.ids.mapping import PipelineCost, assign_targets

        mock_call_llm.return_value = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[],
        )
        cost = PipelineCost()

        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "semantic": sample_subtree,
        }

        assign_targets("jet", "pf_active", context, cost=cost)

        # Inspect the messages passed to _call_llm
        call_args = mock_call_llm.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "TargetAssignmentBatch" in messages[0]["content"]
        assert "jet" in messages[1]["content"]


class TestMapSignalsUsesSystemPrompt:
    """Verify map_signals sends system+user messages with stable system content."""

    @patch("imas_codex.ids.mapping.fetch_source_code_refs")
    @patch("imas_codex.ids.mapping._call_llm")
    @patch("imas_codex.ids.mapping.fetch_imas_fields")
    @patch("imas_codex.ids.mapping.fetch_imas_subtree")
    def test_system_content_identical_across_sections(
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

        batch = SignalMappingBatch(
            ids_name="pf_active",
            target_path="pf_active/coil",
            mappings=[],
        )
        mock_fields.return_value = sample_subtree[1:]
        mock_subtree.return_value = sample_subtree[1:]
        mock_call_llm.return_value = batch
        mock_code_refs.return_value = []
        cost = PipelineCost()

        # Two different source assignments to the same section
        assignments = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_target_path="pf_active/coil",
                    confidence=0.95,
                    reasoning="PF coil 1",
                ),
                TargetAssignment(
                    source_id="jet:pf_coils:group2",
                    imas_target_path="pf_active/coil",
                    confidence=0.90,
                    reasoning="PF coil 2",
                ),
            ],
        )

        groups2 = sample_groups + [
            {**sample_groups[0], "id": "jet:pf_coils:group2", "group_key": "pf_coil_2"}
        ]
        context = {
            "groups": groups2,
            "subtree": sample_subtree,
            "cocos_paths": [],
            "existing": {"mapping": None, "sections": [], "bindings": []},
        }

        map_signals("jet", "pf_active", assignments, context, gc=mock_gc, cost=cost)

        # Both calls should have the SAME system message (cache-optimal)
        assert mock_call_llm.call_count == 2
        call1_msgs = mock_call_llm.call_args_list[0][0][0]
        call2_msgs = mock_call_llm.call_args_list[1][0][0]
        assert call1_msgs[0]["content"] == call2_msgs[0]["content"]
        assert call1_msgs[0]["role"] == "system"
        # User messages differ (different source context)
        assert call1_msgs[1]["content"] != call2_msgs[1]["content"]

    @patch(
        "imas_codex.tools.version_tool.VersionTool.get_dd_version_context",
        new_callable=AsyncMock,
    )
    @patch("imas_codex.ids.mapping.fetch_source_code_refs")
    @patch("imas_codex.ids.mapping._call_llm")
    @patch("imas_codex.ids.mapping.fetch_imas_fields")
    @patch("imas_codex.ids.mapping.fetch_imas_subtree")
    def test_user_prompt_includes_version_not_found_and_no_change_context(
        self,
        mock_subtree,
        mock_fields,
        mock_call_llm,
        mock_code_refs,
        mock_version_context,
        mock_gc,
        sample_groups,
        sample_subtree,
    ):
        from imas_codex.ids.mapping import PipelineCost, map_signals

        mock_fields.return_value = sample_subtree[1:]
        mock_subtree.return_value = sample_subtree[1:]
        mock_call_llm.return_value = SignalMappingBatch(
            ids_name="pf_active",
            target_path="pf_active/coil",
            mappings=[],
        )
        mock_code_refs.return_value = []
        mock_version_context.return_value = {
            "paths": {
                "pf_active/coil/element/geometry/rectangle/r": {
                    "change_count": 0,
                    "notable_changes": [],
                }
            },
            "paths_without_changes": ["pf_active/coil/element/geometry/rectangle/r"],
            "not_found": ["pf_active/coil/element/geometry/rectangle/missing"],
        }
        cost = PipelineCost()

        assignments = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_target_path="pf_active/coil",
                    confidence=0.95,
                    reasoning="PF coil 1",
                ),
            ],
        )
        context = {
            "groups": sample_groups,
            "subtree": sample_subtree,
            "cocos_paths": [],
            "existing": {"mapping": None, "sections": [], "bindings": []},
        }

        map_signals("jet", "pf_active", assignments, context, gc=mock_gc, cost=cost)

        messages = mock_call_llm.call_args[0][0]
        user_prompt = messages[1]["content"]
        assert "Paths without notable changes" in user_prompt
        assert "pf_active/coil/element/geometry/rectangle/r" in user_prompt
        assert "Paths not found in DD graph" in user_prompt
        assert "pf_active/coil/element/geometry/rectangle/missing" in user_prompt


class TestDiscoverAssemblyUsesSystemPrompt:
    """Verify discover_assembly sends system+user messages."""

    @patch("imas_codex.ids.mapping.fetch_imas_fields")
    @patch("imas_codex.ids.mapping.fetch_imas_subtree")
    @patch("imas_codex.ids.mapping._call_llm")
    def test_messages_have_system_role(
        self,
        mock_call_llm,
        mock_subtree,
        mock_fields,
        mock_gc,
        sample_groups,
        sample_subtree,
    ):
        from imas_codex.ids.mapping import PipelineCost, discover_assembly

        mock_subtree.return_value = sample_subtree
        mock_fields.return_value = sample_subtree
        mock_call_llm.return_value = AssemblyConfig(
            target_path="pf_active/coil",
            pattern=AssemblyPattern.ARRAY_PER_NODE,
            confidence=0.85,
            reasoning="One coil per entry",
        )
        cost = PipelineCost()

        assignments = TargetAssignmentBatch(
            ids_name="pf_active",
            assignments=[
                TargetAssignment(
                    source_id="jet:pf_coils:group1",
                    imas_target_path="pf_active/coil",
                    confidence=0.95,
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
                    confidence=0.95,
                    reasoning="R position",
                ),
            ],
        )
        context = {"groups": sample_groups, "subtree": sample_subtree}

        discover_assembly(
            "jet",
            "pf_active",
            assignments,
            [field_batch],
            context,
            gc=mock_gc,
            cost=cost,
        )

        call_args = mock_call_llm.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "array_per_node" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "jet" in messages[1]["content"]


# ---------------------------------------------------------------------------
# 8. DuplicateTargetClassification
# ---------------------------------------------------------------------------


class TestDuplicateTargetClassification:
    def test_all_values(self):
        from imas_codex.ids.validation import DuplicateTargetClassification

        values = set(DuplicateTargetClassification)
        expected = {
            "epoch_variants",
            "processing_stages",
            "redundant_diagnostics",
            "legitimate_other",
            "erroneous",
        }
        assert {v.value for v in values} == expected


# ---------------------------------------------------------------------------
# 9. many_to_one_note on SignalMappingEntry
# ---------------------------------------------------------------------------


class TestManyToOneNote:
    def test_default_none(self):
        e = SignalMappingEntry(
            source_id="s1",
            target_id="t1",
            transform_expression="value",
            confidence=0.9,
        )
        assert e.many_to_one_note is None

    def test_set_note(self):
        e = SignalMappingEntry(
            source_id="s1",
            target_id="t1",
            transform_expression="value",
            confidence=0.9,
            many_to_one_note="Epoch variant: same coil from 2019 campaign",
        )
        assert "Epoch variant" in e.many_to_one_note

    def test_serialization(self):
        e = SignalMappingEntry(
            source_id="s1",
            target_id="t1",
            transform_expression="value",
            confidence=0.9,
            many_to_one_note="Processing stage: subsampled data",
        )
        data = e.model_dump()
        assert data["many_to_one_note"] == "Processing stage: subsampled data"
