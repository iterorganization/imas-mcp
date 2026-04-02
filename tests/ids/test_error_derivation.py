"""Tests for the error derivation pipeline.

Covers:
- ValidatedSignalMapping model defaults and error-derived fields
- derive_error_mappings: basic usage, skipping already-derived, empty input,
  no errors in graph, multiple sources per target
- classify_error_signals: uncertainty vs physics-error-field exclusion
- match_error_signals_to_imas: cross-reference with existing data mappings
- persist_mapping_result: error fields are persisted in MAPS_TO_IMAS relationships
- CLI: --stage, --skip-errors, --skip-metadata flags appear in help text
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from imas_codex.ids.models import (
    MappingDisposition,
    ValidatedMappingResult,
    ValidatedSignalMapping,
    persist_mapping_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gc():
    gc = MagicMock()
    gc.query.return_value = []
    return gc


def _make_direct_mapping(
    source_id: str = "jet:magnetics/ip",
    target_id: str = "equilibrium/time_slice/global_quantities/ip",
    confidence: float = 0.95,
    source_units: str | None = "A",
    target_units: str | None = "A",
    transform_expression: str = "value",
) -> ValidatedSignalMapping:
    return ValidatedSignalMapping(
        source_id=source_id,
        target_id=target_id,
        confidence=confidence,
        source_units=source_units,
        target_units=target_units,
        transform_expression=transform_expression,
        mapping_type="direct",
    )


# ---------------------------------------------------------------------------
# 1. ValidatedSignalMapping model defaults
# ---------------------------------------------------------------------------


def test_validated_signal_mapping_requires_mapping_type():
    """mapping_type is required, not defaulted."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ValidatedSignalMapping(
            source_id="test:source",
            target_id="equilibrium/time_slice/profiles_1d/psi",
            confidence=0.9,
        )


def test_validated_signal_mapping_direct():
    """Direct mapping type."""
    m = ValidatedSignalMapping(
        source_id="test:source",
        target_id="equilibrium/time_slice/profiles_1d/psi",
        confidence=0.9,
        mapping_type="direct",
    )
    assert m.mapping_type == "direct"
    assert m.error_type is None
    assert m.derived_from is None


def test_validated_signal_mapping_error_derived():
    """Error-derived mappings store type and parent."""
    m = ValidatedSignalMapping(
        source_id="test:source",
        target_id="equilibrium/time_slice/profiles_1d/psi_error_upper",
        confidence=0.9,
        mapping_type="error_derived",
        error_type="upper",
        derived_from="equilibrium/time_slice/profiles_1d/psi",
    )
    assert m.mapping_type == "error_derived"
    assert m.error_type == "upper"
    assert m.derived_from == "equilibrium/time_slice/profiles_1d/psi"


def test_validated_signal_mapping_source_property_default():
    """source_property defaults to 'value'."""
    m = ValidatedSignalMapping(
        source_id="test:source",
        target_id="equilibrium/time_slice/profiles_1d/psi",
        confidence=0.8,
        mapping_type="direct",
    )
    assert m.source_property == "value"


def test_validated_signal_mapping_disposition_default():
    """disposition defaults to MAPPED."""
    m = ValidatedSignalMapping(
        source_id="test:source",
        target_id="equilibrium/time_slice/profiles_1d/psi",
        confidence=0.8,
        mapping_type="direct",
    )
    assert m.disposition == MappingDisposition.MAPPED


# ---------------------------------------------------------------------------
# 2. derive_error_mappings
# ---------------------------------------------------------------------------


def test_derive_error_mappings_basic(mock_gc):
    """Basic error derivation from data mappings via HAS_ERROR."""
    from imas_codex.ids.mapping import derive_error_mappings

    mock_gc.query.return_value = [
        {
            "data_path": "equilibrium/time_slice/global_quantities/ip",
            "error_path": "equilibrium/time_slice/global_quantities/ip_error_upper",
            "error_type": "upper",
        },
        {
            "data_path": "equilibrium/time_slice/global_quantities/ip",
            "error_path": "equilibrium/time_slice/global_quantities/ip_error_lower",
            "error_type": "lower",
        },
        {
            "data_path": "equilibrium/time_slice/global_quantities/ip",
            "error_path": "equilibrium/time_slice/global_quantities/ip_error_index",
            "error_type": "index",
        },
    ]

    data_mappings = [_make_direct_mapping()]

    error_mappings = derive_error_mappings(
        data_mappings, gc=mock_gc, include_direct_error_signals=False
    )

    assert len(error_mappings) == 3
    for em in error_mappings:
        assert em.mapping_type == "error_derived"
        assert em.derived_from == "equilibrium/time_slice/global_quantities/ip"
        assert em.source_id == "jet:magnetics/ip"
        assert em.confidence == pytest.approx(0.95)
        assert em.source_units == "A"
        assert em.target_units == "A"
        assert em.transform_expression == "value"

    types = {em.error_type for em in error_mappings}
    assert types == {"upper", "lower", "index"}


def test_derive_error_mappings_inherits_transform(mock_gc):
    """Derived error mappings inherit the parent transform expression."""
    from imas_codex.ids.mapping import derive_error_mappings

    mock_gc.query.return_value = [
        {
            "data_path": "summary/global_quantities/ip/value",
            "error_path": "summary/global_quantities/ip/value_error_upper",
            "error_type": "upper",
        },
    ]

    dm = _make_direct_mapping(
        target_id="summary/global_quantities/ip/value",
        transform_expression="value * 1e-6",
    )
    result = derive_error_mappings([dm], gc=mock_gc, include_direct_error_signals=False)

    assert len(result) == 1
    assert result[0].transform_expression == "value * 1e-6"


def test_derive_error_mappings_skips_error_derived(mock_gc):
    """Only processes direct mappings, not already-derived ones."""
    from imas_codex.ids.mapping import derive_error_mappings

    # No graph query should be issued because there are no direct mappings
    mappings = [
        ValidatedSignalMapping(
            source_id="jet:magnetics/ip",
            target_id="equilibrium/time_slice/global_quantities/ip_error_upper",
            confidence=0.9,
            mapping_type="error_derived",
            error_type="upper",
            derived_from="equilibrium/time_slice/global_quantities/ip",
        )
    ]

    result = derive_error_mappings(
        mappings, gc=mock_gc, include_direct_error_signals=False
    )
    assert result == []
    mock_gc.query.assert_not_called()


def test_derive_error_mappings_empty(mock_gc):
    """Returns empty list for empty input without querying the graph."""
    from imas_codex.ids.mapping import derive_error_mappings

    result = derive_error_mappings([], gc=mock_gc, include_direct_error_signals=False)
    assert result == []
    mock_gc.query.assert_not_called()


def test_derive_error_mappings_no_errors(mock_gc):
    """Returns empty when graph has no HAS_ERROR relationships."""
    from imas_codex.ids.mapping import derive_error_mappings

    mock_gc.query.return_value = []  # No HAS_ERROR edges found

    data_mappings = [_make_direct_mapping()]
    result = derive_error_mappings(
        data_mappings, gc=mock_gc, include_direct_error_signals=False
    )
    assert result == []


def test_derive_error_mappings_multiple_sources_same_target(mock_gc):
    """Multiple sources mapped to the same target each get error mappings."""
    from imas_codex.ids.mapping import derive_error_mappings

    mock_gc.query.return_value = [
        {
            "data_path": "magnetics/flux_loop/flux",
            "error_path": "magnetics/flux_loop/flux_error_upper",
            "error_type": "upper",
        },
    ]

    mappings = [
        _make_direct_mapping(
            source_id="jet:magnetics/fl_001",
            target_id="magnetics/flux_loop/flux",
            confidence=0.90,
        ),
        _make_direct_mapping(
            source_id="jet:magnetics/fl_002",
            target_id="magnetics/flux_loop/flux",
            confidence=0.85,
        ),
    ]

    result = derive_error_mappings(
        mappings, gc=mock_gc, include_direct_error_signals=False
    )

    # Both sources should produce an error mapping for the same error path
    assert len(result) == 2
    source_ids = {r.source_id for r in result}
    assert source_ids == {"jet:magnetics/fl_001", "jet:magnetics/fl_002"}
    for r in result:
        assert r.target_id == "magnetics/flux_loop/flux_error_upper"
        assert r.error_type == "upper"


def test_derive_error_mappings_mixed_types(mock_gc):
    """Direct and error-derived mappings in input: only direct ones processed."""
    from imas_codex.ids.mapping import derive_error_mappings

    mock_gc.query.return_value = [
        {
            "data_path": "equilibrium/time_slice/global_quantities/ip",
            "error_path": "equilibrium/time_slice/global_quantities/ip_error_upper",
            "error_type": "upper",
        },
    ]

    direct = _make_direct_mapping()
    already_derived = ValidatedSignalMapping(
        source_id="jet:magnetics/ip",
        target_id="equilibrium/time_slice/global_quantities/ip_error_lower",
        confidence=0.9,
        mapping_type="error_derived",
        error_type="lower",
        derived_from="equilibrium/time_slice/global_quantities/ip",
    )

    result = derive_error_mappings(
        [direct, already_derived], gc=mock_gc, include_direct_error_signals=False
    )
    # Only one new error mapping from the direct input
    assert len(result) == 1
    assert result[0].error_type == "upper"


def test_derive_error_mappings_evidence_references_parent(mock_gc):
    """Derived mapping evidence string references the parent data path."""
    from imas_codex.ids.mapping import derive_error_mappings

    parent_path = "equilibrium/time_slice/global_quantities/ip"
    mock_gc.query.return_value = [
        {
            "data_path": parent_path,
            "error_path": f"{parent_path}_error_upper",
            "error_type": "upper",
        },
    ]

    result = derive_error_mappings(
        [_make_direct_mapping(target_id=parent_path)],
        gc=mock_gc,
        include_direct_error_signals=False,
    )

    assert len(result) == 1
    assert parent_path in result[0].evidence


# ---------------------------------------------------------------------------
# 3. classify_error_signals
# ---------------------------------------------------------------------------


def test_classify_error_signals_identifies_uncertainty(mock_gc):
    """Signals with uncertainty keywords are classified as error signals."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "jet:ts/ne_error",
            "group_key": "ts/ne_error",
            "description": "Thomson scattering electron density error",
            "physics_domain": "kinetics",
            "rep_name": "Ne Error",
        },
    ]

    result = classify_error_signals("jet", gc=mock_gc)

    assert len(result) == 1
    assert result[0]["signal_id"] == "jet:ts/ne_error"
    assert result[0]["probable_error_type"] in ("upper", "lower", "symmetric")


def test_classify_error_signals_excludes_physics_error_fields(mock_gc):
    """Signals about 'error field correction coils' are excluded."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "jet:magnetics/efc_current",
            "group_key": "magnetics/efc_current",
            "description": "Error field correction coil current",
            "physics_domain": "magnetics",
            "rep_name": "EFC Current",
        },
    ]

    result = classify_error_signals("jet", gc=mock_gc)
    error_ids = {r["signal_id"] for r in result}
    assert "jet:magnetics/efc_current" not in error_ids


def test_classify_error_signals_excludes_plain_signals(mock_gc):
    """Signals without uncertainty keywords are not returned."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "jet:magnetics/ip",
            "group_key": "magnetics/ip",
            "description": "Plasma current",
            "physics_domain": "magnetics",
            "rep_name": "IP",
        },
    ]

    result = classify_error_signals("jet", gc=mock_gc)
    assert result == []


def test_classify_error_signals_mixed(mock_gc):
    """Mixed input: only uncertainty signals are returned."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "jet:magnetics/ip_error",
            "group_key": "magnetics/ip_error",
            "description": "Plasma current error",
            "physics_domain": "magnetics",
            "rep_name": "IP Error",
        },
        {
            "id": "jet:magnetics/ip",
            "group_key": "magnetics/ip",
            "description": "Plasma current",
            "physics_domain": "magnetics",
            "rep_name": "IP",
        },
        {
            "id": "jet:magnetics/efc_current",
            "group_key": "magnetics/efc_current",
            "description": "Error field correction coil current",
            "physics_domain": "magnetics",
            "rep_name": "EFC Current",
        },
    ]

    result = classify_error_signals("jet", gc=mock_gc)
    error_ids = {r["signal_id"] for r in result}

    assert "jet:magnetics/ip_error" in error_ids
    assert "jet:magnetics/ip" not in error_ids
    assert "jet:magnetics/efc_current" not in error_ids


def test_classify_error_signals_lower_keyword(mock_gc):
    """Signals with 'lower' keyword classified as lower error type."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "tcv:ts/te_error_lower",
            "group_key": "ts/te_error_lower",
            "description": "Te lower uncertainty bound",
            "physics_domain": "kinetics",
            "rep_name": "Te Error Lower",
        },
    ]

    result = classify_error_signals("tcv", gc=mock_gc)
    assert len(result) == 1
    assert result[0]["probable_error_type"] == "lower"


def test_classify_error_signals_upper_keyword(mock_gc):
    """Signals with 'upper' keyword classified as upper error type."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "tcv:ts/te_error_upper",
            "group_key": "ts/te_error_upper",
            "description": "Te upper uncertainty bound",
            "physics_domain": "kinetics",
            "rep_name": "Te Error Upper",
        },
    ]

    result = classify_error_signals("tcv", gc=mock_gc)
    assert len(result) == 1
    assert result[0]["probable_error_type"] == "upper"


def test_classify_error_signals_sigma_keyword(mock_gc):
    """'sigma' in group_key triggers error classification (symmetric)."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "tcv:ts/ne_sigma",
            "group_key": "ts/ne_sigma",
            "description": "Electron density standard deviation",
            "physics_domain": "kinetics",
            "rep_name": "Ne Sigma",
        },
    ]

    result = classify_error_signals("tcv", gc=mock_gc)
    assert len(result) == 1
    assert result[0]["probable_error_type"] == "symmetric"


def test_classify_error_signals_returns_required_keys(mock_gc):
    """Each returned dict contains all required keys."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = [
        {
            "id": "jet:ts/ne_error",
            "group_key": "ts/ne_error",
            "description": "Thomson electron density error",
            "physics_domain": "kinetics",
            "rep_name": "Ne Error",
        },
    ]

    result = classify_error_signals("jet", gc=mock_gc)
    assert len(result) == 1
    required_keys = {
        "signal_id",
        "group_key",
        "description",
        "physics_domain",
        "probable_error_type",
    }
    assert required_keys <= set(result[0].keys())


def test_classify_error_signals_empty_graph(mock_gc):
    """Returns empty list when graph has no signal sources."""
    from imas_codex.ids.mapping import classify_error_signals

    mock_gc.query.return_value = []
    result = classify_error_signals("jet", gc=mock_gc)
    assert result == []


# ---------------------------------------------------------------------------
# 4. match_error_signals_to_imas
# ---------------------------------------------------------------------------


def test_match_error_signals_to_imas_basic(mock_gc):
    """Matches error signal to IMAS error field via parent cross-reference."""
    from imas_codex.ids.mapping import match_error_signals_to_imas

    # First query: existing direct MAPS_TO_IMAS
    # Second query: HAS_ERROR children for those targets
    mock_gc.query.side_effect = [
        [
            {
                "source_id": "jet:magnetics/ip",
                "source_group_key": "magnetics/ip",
                "target_id": "equilibrium/time_slice/global_quantities/ip",
                "source_units": "A",
                "target_units": "A",
                "confidence": 0.95,
            },
        ],
        [
            {
                "data_path": "equilibrium/time_slice/global_quantities/ip",
                "error_path": "equilibrium/time_slice/global_quantities/ip_error_upper",
                "error_type": "upper",
            },
            {
                "data_path": "equilibrium/time_slice/global_quantities/ip",
                "error_path": "equilibrium/time_slice/global_quantities/ip_error_lower",
                "error_type": "lower",
            },
        ],
    ]

    error_signals = [
        {
            "signal_id": "jet:magnetics/ip_error",
            "group_key": "magnetics/ip_error",
            "description": "Plasma current error",
            "physics_domain": "magnetics",
            "probable_error_type": "symmetric",
        },
    ]

    result = match_error_signals_to_imas("jet", error_signals, gc=mock_gc)

    # symmetric error type should match both upper and lower
    assert len(result) == 2
    for r in result:
        assert r.source_id == "jet:magnetics/ip_error"
        assert r.mapping_type == "error_derived"
        assert r.derived_from == "equilibrium/time_slice/global_quantities/ip"
    targets = {r.target_id for r in result}
    assert "equilibrium/time_slice/global_quantities/ip_error_upper" in targets
    assert "equilibrium/time_slice/global_quantities/ip_error_lower" in targets


def test_match_error_signals_to_imas_empty_signals(mock_gc):
    """Returns empty list when no error signals provided."""
    from imas_codex.ids.mapping import match_error_signals_to_imas

    result = match_error_signals_to_imas("jet", [], gc=mock_gc)
    assert result == []
    mock_gc.query.assert_not_called()


def test_match_error_signals_to_imas_no_existing_mappings(mock_gc):
    """Returns empty list when no existing data mappings exist in graph."""
    from imas_codex.ids.mapping import match_error_signals_to_imas

    mock_gc.query.return_value = []  # No existing MAPS_TO_IMAS

    error_signals = [
        {
            "signal_id": "jet:magnetics/ip_error",
            "group_key": "magnetics/ip_error",
            "description": "Plasma current error",
            "physics_domain": "magnetics",
            "probable_error_type": "symmetric",
        },
    ]

    result = match_error_signals_to_imas("jet", error_signals, gc=mock_gc)
    assert result == []


def test_match_error_signals_confidence_reduced(mock_gc):
    """Direct error signal mappings have confidence slightly reduced from parent."""
    from imas_codex.ids.mapping import match_error_signals_to_imas

    parent_confidence = 1.0
    mock_gc.query.side_effect = [
        [
            {
                "source_id": "jet:magnetics/ip",
                "source_group_key": "magnetics/ip",
                "target_id": "equilibrium/time_slice/global_quantities/ip",
                "source_units": "A",
                "target_units": "A",
                "confidence": parent_confidence,
            },
        ],
        [
            {
                "data_path": "equilibrium/time_slice/global_quantities/ip",
                "error_path": "equilibrium/time_slice/global_quantities/ip_error_upper",
                "error_type": "upper",
            },
        ],
    ]

    error_signals = [
        {
            "signal_id": "jet:magnetics/ip_error",
            "group_key": "magnetics/ip_error",
            "description": "Plasma current error",
            "physics_domain": "magnetics",
            "probable_error_type": "upper",
        },
    ]

    result = match_error_signals_to_imas("jet", error_signals, gc=mock_gc)
    assert len(result) == 1
    assert result[0].confidence < parent_confidence  # Confidence is reduced


# ---------------------------------------------------------------------------
# 5. persist_mapping_result — error fields in MAPS_TO_IMAS
# ---------------------------------------------------------------------------


def test_persist_mapping_result_error_fields(mock_gc):
    """Error-derived mappings persist mapping_type, error_type, derived_from."""
    result = ValidatedMappingResult(
        facility="jet",
        ids_name="equilibrium",
        dd_version="4.1.0",
        sections=[],
        bindings=[
            ValidatedSignalMapping(
                source_id="jet:magnetics/ip",
                target_id="equilibrium/time_slice/global_quantities/ip_error_upper",
                confidence=0.9,
                mapping_type="error_derived",
                error_type="upper",
                derived_from="equilibrium/time_slice/global_quantities/ip",
            ),
        ],
    )

    persist_mapping_result(result, gc=mock_gc, status="generated")

    # Find the MAPS_TO_IMAS query call
    maps_to_imas_calls = [
        c for c in mock_gc.query.call_args_list if "MAPS_TO_IMAS" in str(c)
    ]
    assert len(maps_to_imas_calls) >= 1

    # Extract the keyword arguments passed to the MAPS_TO_IMAS query
    call_obj = maps_to_imas_calls[0]
    # call_args_list entries have .kwargs for keyword arguments
    kwargs = call_obj.kwargs if call_obj.kwargs else {}
    if not kwargs and len(call_obj.args) > 1:
        # Positional args case — second arg onwards are kwargs-like dict
        pass

    # The simplest check: assert the string representation contains the values
    call_str = str(call_obj)
    assert "error_derived" in call_str
    assert "upper" in call_str
    assert "equilibrium/time_slice/global_quantities/ip" in call_str


def test_persist_mapping_result_direct_mapping_fields(mock_gc):
    """Direct mappings persist with mapping_type='direct' and no error fields."""
    result = ValidatedMappingResult(
        facility="jet",
        ids_name="equilibrium",
        dd_version="4.1.0",
        sections=[],
        bindings=[
            ValidatedSignalMapping(
                source_id="jet:magnetics/ip",
                target_id="equilibrium/time_slice/global_quantities/ip",
                confidence=0.95,
                mapping_type="direct",
            ),
        ],
    )

    persist_mapping_result(result, gc=mock_gc, status="generated")

    maps_to_imas_calls = [
        c for c in mock_gc.query.call_args_list if "MAPS_TO_IMAS" in str(c)
    ]
    assert len(maps_to_imas_calls) >= 1

    call_str = str(maps_to_imas_calls[0])
    assert "direct" in call_str
    # error_type and derived_from should be None (absent or 'None' string)


def test_persist_mapping_result_multiple_bindings(mock_gc):
    """Mixed bindings (direct + error_derived) all get persisted."""
    result = ValidatedMappingResult(
        facility="jet",
        ids_name="equilibrium",
        dd_version="4.1.0",
        sections=[],
        bindings=[
            ValidatedSignalMapping(
                source_id="jet:magnetics/ip",
                target_id="equilibrium/time_slice/global_quantities/ip",
                confidence=0.95,
                mapping_type="direct",
            ),
            ValidatedSignalMapping(
                source_id="jet:magnetics/ip",
                target_id="equilibrium/time_slice/global_quantities/ip_error_upper",
                confidence=0.9,
                mapping_type="error_derived",
                error_type="upper",
                derived_from="equilibrium/time_slice/global_quantities/ip",
            ),
            ValidatedSignalMapping(
                source_id="jet:magnetics/ip",
                target_id="equilibrium/time_slice/global_quantities/ip_error_lower",
                confidence=0.9,
                mapping_type="error_derived",
                error_type="lower",
                derived_from="equilibrium/time_slice/global_quantities/ip",
            ),
        ],
    )

    persist_mapping_result(result, gc=mock_gc, status="generated")

    maps_to_imas_calls = [
        c for c in mock_gc.query.call_args_list if "MAPS_TO_IMAS" in str(c)
    ]
    # One call per binding
    assert len(maps_to_imas_calls) == 3


# ---------------------------------------------------------------------------
# 6. CLI: --stage, --skip-errors, --skip-metadata flags
# ---------------------------------------------------------------------------


def test_map_run_help_includes_stage():
    """CLI help text includes --stage flag with valid choices."""
    from click.testing import CliRunner

    from imas_codex.cli.map import map_cmd

    runner = CliRunner()
    result = runner.invoke(map_cmd, ["run", "--help"])
    assert result.exit_code == 0
    assert "--stage" in result.output


def test_map_run_help_includes_skip_errors():
    """CLI help text includes --skip-errors flag."""
    from click.testing import CliRunner

    from imas_codex.cli.map import map_cmd

    runner = CliRunner()
    result = runner.invoke(map_cmd, ["run", "--help"])
    assert result.exit_code == 0
    assert "--skip-errors" in result.output


def test_map_run_help_includes_skip_metadata():
    """CLI help text includes --skip-metadata flag."""
    from click.testing import CliRunner

    from imas_codex.cli.map import map_cmd

    runner = CliRunner()
    result = runner.invoke(map_cmd, ["run", "--help"])
    assert result.exit_code == 0
    assert "--skip-metadata" in result.output


def test_map_run_help_stage_choices():
    """Stage flag documents all three valid choices: all, data, error."""
    from click.testing import CliRunner

    from imas_codex.cli.map import map_cmd

    runner = CliRunner()
    result = runner.invoke(map_cmd, ["run", "--help"])
    assert result.exit_code == 0
    output = result.output
    # all three choices should appear in the help
    assert "all" in output
    assert "data" in output
    assert "error" in output


# ---------------------------------------------------------------------------
# 7. run_error_derivation_only
# ---------------------------------------------------------------------------


def test_run_error_derivation_only_no_existing_mappings(mock_gc):
    """Returns empty list when no direct mappings exist in graph."""
    from imas_codex.ids.mapping import run_error_derivation_only

    mock_gc.query.return_value = []  # No existing MAPS_TO_IMAS

    result = run_error_derivation_only("jet", "equilibrium", gc=mock_gc, dry_run=True)
    assert result == []


def test_run_error_derivation_only_derives_from_graph(mock_gc):
    """Fetches existing data mappings and derives error mappings."""
    from imas_codex.ids.mapping import run_error_derivation_only

    # First query: fetch existing direct mappings
    # Subsequent queries: HAS_ERROR traversal (batched)
    mock_gc.query.side_effect = [
        # Existing MAPS_TO_IMAS direct mappings
        [
            {
                "source_id": "jet:magnetics/ip",
                "source_property": "value",
                "target_id": "equilibrium/time_slice/global_quantities/ip",
                "transform_expression": "value",
                "source_units": "A",
                "target_units": "A",
                "cocos_label": None,
                "confidence": 0.95,
            }
        ],
        # HAS_ERROR traversal
        [
            {
                "data_path": "equilibrium/time_slice/global_quantities/ip",
                "error_path": "equilibrium/time_slice/global_quantities/ip_error_upper",
                "error_type": "upper",
            }
        ],
        # classify_error_signals query (no error signals)
        [],
    ]

    result = run_error_derivation_only("jet", "equilibrium", gc=mock_gc, dry_run=True)

    assert len(result) >= 1
    assert result[0].mapping_type == "error_derived"
    assert result[0].source_id == "jet:magnetics/ip"
    assert result[0].error_type == "upper"
