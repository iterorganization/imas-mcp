"""Tests for imas_codex.ids.validation module."""

from unittest.mock import MagicMock

import pytest

from imas_codex.ids.models import ValidatedSignalMapping
from imas_codex.ids.validation import (
    AssemblyCoverageReport,
    BindingCheck,
    ConfidenceDistribution,
    CoverageReport,
    SignalCoverageReport,
    SignalSourceCoverageReport,
    ValidationReport,
    compute_assembly_coverage,
    compute_confidence_distribution,
    compute_coverage,
    compute_signal_coverage,
    compute_signal_source_coverage,
    validate_mapping,
)


@pytest.fixture
def mock_gc():
    return MagicMock()


def _make_binding(**overrides):
    defaults = {
        "source_id": "jet:ids:pf_active:pf_coil_1",
        "source_property": "value",
        "target_id": "pf_active/coil/element/geometry/rectangle/r",
        "transform_expression": "value",
        "source_units": None,
        "target_units": None,
        "cocos_label": None,
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return ValidatedSignalMapping(**defaults)


class TestValidateMapping:
    def test_empty_bindings(self, mock_gc):
        report = validate_mapping([], gc=mock_gc)
        assert report.all_passed is True
        assert report.binding_checks == []
        assert report.duplicate_targets == []

    def test_all_checks_pass(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding()
        # check_imas_paths mock — return path exists
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [
                {
                    "path": "pf_active/coil/element/geometry/rectangle/r",
                    "exists": True,
                    "data_type": "FLT_0D",
                    "units": "m",
                }
            ]
            report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is True
        assert len(report.binding_checks) == 1
        check = report.binding_checks[0]
        assert check.source_exists is True
        assert check.target_exists is True
        assert check.transform_executes is True
        assert check.units_compatible is True
        assert check.error is None

    def test_source_not_found(self, mock_gc):
        # SignalSource query returns empty
        mock_gc.query.return_value = []

        binding = _make_binding()
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [{"path": binding.target_id, "exists": True}]
            report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is False
        assert report.binding_checks[0].source_exists is False
        assert "not found" in report.binding_checks[0].error

    def test_target_not_found(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding(target_id="pf_active/nonexistent/field")
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [
                {"path": "pf_active/nonexistent/field", "exists": False}
            ]
            report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is False
        assert report.binding_checks[0].target_exists is False

    def test_target_renamed(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding(target_id="pf_active/old_path")
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [
                {
                    "path": "pf_active/old_path",
                    "exists": False,
                    "suggestion": "pf_active/new_path",
                }
            ]
            report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is False
        assert "renamed" in report.binding_checks[0].error

    def test_transform_failure(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding(transform_expression="invalid_syntax(")
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [{"path": binding.target_id, "exists": True}]
            report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is False
        assert report.binding_checks[0].transform_executes is False
        assert "Transform" in report.binding_checks[0].error

    def test_units_incompatible(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding(source_units="m", target_units="kg")
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [{"path": binding.target_id, "exists": True}]
            with patch("imas_codex.ids.validation.analyze_units") as mock_units:
                mock_units.return_value = {"compatible": False}
                report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is False
        assert report.binding_checks[0].units_compatible is False

    def test_units_compatible(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding(source_units="mm", target_units="m")
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [{"path": binding.target_id, "exists": True}]
            with patch("imas_codex.ids.validation.analyze_units") as mock_units:
                mock_units.return_value = {
                    "compatible": True,
                    "conversion_factor": 0.001,
                }
                report = validate_mapping([binding], gc=mock_gc)

        assert report.all_passed is True
        assert report.binding_checks[0].units_compatible is True

    def test_no_units_is_compatible(self, mock_gc):
        mock_gc.query.return_value = [{"id": "jet:ids:pf_active:pf_coil_1"}]

        binding = _make_binding(source_units=None, target_units=None)
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [{"path": binding.target_id, "exists": True}]
            report = validate_mapping([binding], gc=mock_gc)

        assert report.binding_checks[0].units_compatible is True

    def test_duplicate_targets(self, mock_gc):
        """Multiple sources → same target with different physics domains = erroneous."""

        def mock_query(cypher, **kwargs):
            # Source existence check
            if "SignalSource" in cypher and "RETURN sg.id AS id LIMIT 1" in cypher:
                return [{"id": kwargs.get("id", "some_group")}]
            # Classification query for many-to-one
            if "UNWIND" in cypher and "group_key" in cypher:
                return [
                    {
                        "id": "group_a",
                        "group_key": "PF:r",
                        "physics_domain": "magnetic_field_systems",
                        "description": "Coil radius",
                    },
                    {
                        "id": "group_b",
                        "group_key": "MP:angle",
                        "physics_domain": "plasma_diagnostics",
                        "description": "Probe angle",
                    },
                ]
            return [{"id": kwargs.get("id", "some_group")}]

        mock_gc.query.side_effect = mock_query

        b1 = _make_binding(
            source_id="group_a", target_id="pf_active/circuit/description"
        )
        b2 = _make_binding(
            source_id="group_b", target_id="pf_active/circuit/description"
        )
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [
                {"path": "pf_active/circuit/description", "exists": True}
            ]
            report = validate_mapping([b1, b2], gc=mock_gc)

        assert "pf_active/circuit/description" in report.duplicate_targets
        assert report.all_passed is False
        # Should have escalation for erroneous many-to-one
        dup_escalations = [
            e
            for e in report.escalations
            if "erroneous" in e.reason.lower() or "many-to-one" in e.reason.lower()
        ]
        assert len(dup_escalations) == 1

    def test_duplicate_targets_legitimate(self, mock_gc):
        """Multiple sources → same target with same physics domain = legitimate."""

        def mock_query(cypher, **kwargs):
            if "SignalSource" in cypher and "RETURN sg.id AS id LIMIT 1" in cypher:
                return [{"id": kwargs.get("id", "some_group")}]
            if "UNWIND" in cypher and "group_key" in cypher:
                return [
                    {
                        "id": "group_a",
                        "group_key": "PF:r_v1",
                        "physics_domain": "magnetic_field_systems",
                        "description": "Coil R v1",
                    },
                    {
                        "id": "group_b",
                        "group_key": "PF:r_v2",
                        "physics_domain": "magnetic_field_systems",
                        "description": "Coil R v2",
                    },
                ]
            return [{"id": kwargs.get("id", "some_group")}]

        mock_gc.query.side_effect = mock_query

        b1 = _make_binding(
            source_id="group_a", target_id="pf_active/circuit/description"
        )
        b2 = _make_binding(
            source_id="group_b", target_id="pf_active/circuit/description"
        )
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [
                {"path": "pf_active/circuit/description", "exists": True}
            ]
            report = validate_mapping([b1, b2], gc=mock_gc)

        # Legitimate many-to-one should NOT be in duplicate_targets
        assert "pf_active/circuit/description" not in report.duplicate_targets
        assert report.all_passed is True

    def test_multiple_bindings_mixed(self, mock_gc):
        """One good binding, one with missing target."""
        # Source query — both exist
        mock_gc.query.return_value = [{"id": "some_group"}]

        good = _make_binding(
            source_id="group_a",
            target_id="pf_active/coil/name",
        )
        bad = _make_binding(
            source_id="group_a",
            target_id="pf_active/nonexistent",
        )
        from unittest.mock import patch

        with patch("imas_codex.ids.validation.check_imas_paths") as mock_check:
            mock_check.return_value = [
                {"path": "pf_active/coil/name", "exists": True},
                {"path": "pf_active/nonexistent", "exists": False},
            ]
            report = validate_mapping([good, bad], gc=mock_gc)

        assert report.all_passed is False
        assert report.binding_checks[0].target_exists is True
        assert report.binding_checks[1].target_exists is False


class TestBindingCheck:
    def test_defaults(self):
        check = BindingCheck(source_id="a", target_id="b")
        assert check.source_exists is False
        assert check.target_exists is False
        assert check.transform_executes is False
        assert check.units_compatible is False
        assert check.error is None


class TestValidationReport:
    def test_defaults(self):
        report = ValidationReport(mapping_id="test:mapping")
        assert report.all_passed is False
        assert report.binding_checks == []
        assert report.duplicate_targets == []
        assert report.escalations == []


class TestComputeCoverage:
    def test_empty_bindings(self, mock_gc):
        mock_gc.query.return_value = [
            {"id": "pf_active/coil/name"},
            {"id": "pf_active/coil/element/geometry/rectangle/r"},
        ]
        report = compute_coverage("pf_active", [], gc=mock_gc)
        assert report.total_leaf_fields == 2
        assert report.mapped_fields == 0
        assert report.percentage == 0.0
        assert len(report.unmapped_fields) == 2

    def test_partial_coverage(self, mock_gc):
        mock_gc.query.return_value = [
            {"id": "pf_active/coil/name"},
            {"id": "pf_active/coil/element/geometry/rectangle/r"},
            {"id": "pf_active/coil/element/geometry/rectangle/z"},
        ]
        bindings = [
            _make_binding(target_id="pf_active/coil/name"),
            _make_binding(target_id="pf_active/coil/element/geometry/rectangle/r"),
        ]
        report = compute_coverage("pf_active", bindings, gc=mock_gc)
        assert report.total_leaf_fields == 3
        assert report.mapped_fields == 2
        assert report.percentage == pytest.approx(66.67, abs=0.1)
        assert "pf_active/coil/element/geometry/rectangle/z" in report.unmapped_fields

    def test_full_coverage(self, mock_gc):
        mock_gc.query.return_value = [{"id": "pf_active/coil/name"}]
        bindings = [_make_binding(target_id="pf_active/coil/name")]
        report = compute_coverage("pf_active", bindings, gc=mock_gc)
        assert report.total_leaf_fields == 1
        assert report.mapped_fields == 1
        assert report.percentage == 100.0
        assert report.unmapped_fields == []

    def test_no_leaf_fields(self, mock_gc):
        mock_gc.query.return_value = []
        report = compute_coverage("pf_active", [], gc=mock_gc)
        assert report.total_leaf_fields == 0
        assert report.percentage == 0.0


class TestCoverageReport:
    def test_defaults(self):
        report = CoverageReport(ids_name="pf_active")
        assert report.total_leaf_fields == 0
        assert report.mapped_fields == 0
        assert report.percentage == 0.0
        assert report.unmapped_fields == []
        assert report.mapped_paths == []


class TestSignalCoverageReport:
    def test_defaults(self):
        report = SignalCoverageReport(facility="jet")
        assert report.total_enriched == 0
        assert report.mapped == 0
        assert report.percentage == 0.0
        assert report.unmapped_groups == []


class TestComputeSignalCoverage:
    def test_no_enriched_groups(self, mock_gc):
        mock_gc.query.return_value = []
        report = compute_signal_coverage("jet", gc=mock_gc)
        assert report.total_enriched == 0
        assert report.mapped == 0
        assert report.percentage == 0.0
        assert report.unmapped_groups == []

    def test_all_mapped(self, mock_gc):
        mock_gc.query.return_value = [
            {"id": "jet:ids:pf_active:coil_1", "is_mapped": True},
            {"id": "jet:ids:pf_active:coil_2", "is_mapped": True},
        ]
        report = compute_signal_coverage("jet", gc=mock_gc)
        assert report.total_enriched == 2
        assert report.mapped == 2
        assert report.percentage == 100.0
        assert report.unmapped_groups == []

    def test_partial_mapped(self, mock_gc):
        mock_gc.query.return_value = [
            {"id": "jet:ids:pf_active:coil_1", "is_mapped": True},
            {"id": "jet:ids:pf_active:coil_2", "is_mapped": False},
            {"id": "jet:ids:pf_active:coil_3", "is_mapped": False},
        ]
        report = compute_signal_coverage("jet", gc=mock_gc)
        assert report.total_enriched == 3
        assert report.mapped == 1
        assert report.percentage == pytest.approx(33.33, abs=0.1)
        assert report.unmapped_groups == [
            "jet:ids:pf_active:coil_2",
            "jet:ids:pf_active:coil_3",
        ]

    def test_none_mapped(self, mock_gc):
        mock_gc.query.return_value = [
            {"id": "jet:ids:pf_active:coil_1", "is_mapped": False},
        ]
        report = compute_signal_coverage("jet", gc=mock_gc)
        assert report.total_enriched == 1
        assert report.mapped == 0
        assert report.percentage == 0.0
        assert report.unmapped_groups == ["jet:ids:pf_active:coil_1"]


# ---------------------------------------------------------------------------
# Phase 8: Extended coverage metrics
# ---------------------------------------------------------------------------


class TestSignalSourceCoverageReport:
    def test_defaults(self):
        report = SignalSourceCoverageReport(facility="jet", ids_name="pf_active")
        assert report.total_enriched_matching == 0
        assert report.mapped_to_ids == 0
        assert report.discovered_sources == 0
        assert report.multi_target_sources == 0
        assert report.enriched_mapped_pct == 0.0


class TestComputeSignalSourceCoverage:
    def test_no_enriched_sources(self, mock_gc):
        mock_gc.query.return_value = []
        report = compute_signal_source_coverage("jet", "pf_active", gc=mock_gc)
        assert report.total_enriched_matching == 0
        assert report.enriched_mapped_pct == 0.0

    def test_partial_mapped(self, mock_gc):
        mock_gc.query.side_effect = [
            # Enriched query
            [
                {"id": "jet:pf:group1", "is_mapped": True},
                {"id": "jet:pf:group2", "is_mapped": False},
            ],
            # Discovered query
            [{"id": "jet:pf:disc1"}],
            # Multi-target query
            [{"id": "jet:pf:group1", "target_count": 3}],
        ]
        report = compute_signal_source_coverage("jet", "pf_active", gc=mock_gc)
        assert report.total_enriched_matching == 2
        assert report.mapped_to_ids == 1
        assert report.enriched_mapped_pct == pytest.approx(50.0)
        assert report.unmapped_enriched == ["jet:pf:group2"]
        assert report.discovered_sources == 1
        assert report.multi_target_sources == 1

    def test_all_mapped_no_discovered(self, mock_gc):
        mock_gc.query.side_effect = [
            [{"id": "jet:pf:group1", "is_mapped": True}],
            [],  # no discovered
            [],  # no multi-target
        ]
        report = compute_signal_source_coverage("jet", "pf_active", gc=mock_gc)
        assert report.total_enriched_matching == 1
        assert report.mapped_to_ids == 1
        assert report.enriched_mapped_pct == 100.0
        assert report.unmapped_enriched == []
        assert report.discovered_sources == 0


class TestAssemblyCoverageReport:
    def test_defaults(self):
        report = AssemblyCoverageReport(facility="jet", ids_name="pf_active")
        assert report.total_sections == 0
        assert report.sections_with_config == 0
        assert report.default_pattern_count == 0
        assert report.custom_pattern_count == 0


class TestComputeAssemblyCoverage:
    def test_no_sections(self, mock_gc):
        mock_gc.query.side_effect = [[], []]
        report = compute_assembly_coverage("jet", "pf_active", gc=mock_gc)
        assert report.total_sections == 0

    def test_with_config(self, mock_gc):
        mock_gc.query.side_effect = [
            # POPULATES query
            [
                {
                    "section_path": "pf_active/coil",
                    "pattern": "array_per_node",
                    "init_arrays": '{"coil": 12}',
                },
                {
                    "section_path": "pf_active/circuit",
                    "pattern": "concatenate",
                    "init_arrays": None,
                },
            ],
            # Binding sections query
            [{"section_path": "pf_active/coil"}],
        ]
        report = compute_assembly_coverage("jet", "pf_active", gc=mock_gc)
        assert report.total_sections == 2
        assert report.sections_with_config == 2
        assert report.default_pattern_count == 1  # array_per_node
        assert report.custom_pattern_count == 1  # concatenate
        assert report.init_arrays_configured == 1
        assert report.init_arrays_unconfigured == 1
        assert report.sections_without_config == []

    def test_section_without_config(self, mock_gc):
        mock_gc.query.side_effect = [
            # POPULATES query — only one section configured
            [
                {
                    "section_path": "pf_active/coil",
                    "pattern": "array_per_node",
                    "init_arrays": None,
                },
            ],
            # Binding sections — two sections have bindings
            [
                {"section_path": "pf_active/coil"},
                {"section_path": "pf_active/supply"},
            ],
        ]
        report = compute_assembly_coverage("jet", "pf_active", gc=mock_gc)
        assert report.total_sections == 2
        assert report.sections_with_config == 1
        assert report.sections_without_config == ["pf_active/supply"]


class TestConfidenceDistribution:
    def test_defaults(self):
        dist = ConfidenceDistribution()
        assert dist.total_bindings == 0
        assert dist.low_count == 0
        assert dist.medium_count == 0
        assert dist.high_count == 0
        assert dist.average_confidence == 0.0


class TestComputeConfidenceDistribution:
    def test_empty_bindings(self):
        dist = compute_confidence_distribution([])
        assert dist.total_bindings == 0

    def test_all_high(self):
        bindings = [
            _make_binding(confidence=0.95),
            _make_binding(confidence=0.85),
        ]
        dist = compute_confidence_distribution(bindings)
        assert dist.total_bindings == 2
        assert dist.high_count == 2
        assert dist.medium_count == 0
        assert dist.low_count == 0
        assert dist.average_confidence == pytest.approx(0.9)

    def test_mixed_distribution(self):
        bindings = [
            _make_binding(confidence=0.95),  # high
            _make_binding(confidence=0.7, target_id="pf_active/coil/name"),  # medium
            _make_binding(
                confidence=0.3, target_id="pf_active/coil/current/data"
            ),  # low
        ]
        dist = compute_confidence_distribution(bindings)
        assert dist.total_bindings == 3
        assert dist.high_count == 1
        assert dist.medium_count == 1
        assert dist.low_count == 1
        assert len(dist.low_bindings) == 1
        assert "0.30" in dist.low_bindings[0]

    def test_boundary_values(self):
        bindings = [
            _make_binding(confidence=0.5),  # medium (0.5 inclusive)
            _make_binding(
                confidence=0.8, target_id="pf_active/coil/name"
            ),  # medium (0.8 inclusive)
        ]
        dist = compute_confidence_distribution(bindings)
        assert dist.medium_count == 2
        assert dist.low_count == 0
        assert dist.high_count == 0
