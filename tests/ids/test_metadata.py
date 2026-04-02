"""Tests for IDS metadata population (Stage 3 of mapping pipeline)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.ids.metadata import (
    IDSMetadataResult,
    MetadataContext,
    _format_signals_summary,
    build_metadata_context,
    persist_metadata,
    populate_deterministic_fields,
    populate_metadata,
)
from imas_codex.ids.models import MetadataPopulationResponse


class TestMetadataContext:
    """Tests for MetadataContext builder."""

    def test_build_metadata_context_basic(self):
        """Test building context with mocked graph."""
        gc = MagicMock()
        gc.query.return_value = [{"ver": "4.0.0"}]

        ctx = build_metadata_context("jet", "pf_active", gc=gc, dd_version="4.0.0")

        assert ctx.dd_version == "4.0.0"
        assert ctx.provider == "jet"
        assert ctx.source.startswith("imas-codex v")
        assert ctx.creation_date  # non-empty ISO timestamp
        assert isinstance(ctx.library_deps, list)

    def test_build_metadata_context_graph_fallback(self):
        """Test graceful fallback when graph is unavailable."""
        gc = MagicMock()
        gc.query.side_effect = Exception("Neo4j down")

        ctx = build_metadata_context("tcv", "equilibrium", gc=gc, dd_version="4.1.0")

        assert ctx.dd_version == "4.1.0"  # Falls back to input
        assert ctx.provider == "tcv"

    def test_build_metadata_context_no_ddversion_match(self):
        """Test when DDVersion not found in graph."""
        gc = MagicMock()
        gc.query.return_value = []

        ctx = build_metadata_context("jet", "magnetics", gc=gc, dd_version="99.0.0")

        assert ctx.dd_version == "99.0.0"  # Uses input as-is

    def test_build_metadata_context_resolves_version_from_graph(self):
        """Test that graph-resolved DD version is used when available."""
        gc = MagicMock()
        gc.query.return_value = [{"ver": "4.0.1"}]

        ctx = build_metadata_context("jet", "pf_active", gc=gc, dd_version="4.0.1")

        assert ctx.dd_version == "4.0.1"

    def test_build_metadata_context_fields(self):
        """Test that all expected fields are populated on MetadataContext."""
        gc = MagicMock()
        gc.query.return_value = []

        ctx = build_metadata_context(
            "west", "core_profiles", gc=gc, dd_version="3.39.0"
        )

        assert ctx.provider == "west"
        assert ctx.dd_version == "3.39.0"
        assert ctx.access_layer_version  # may be "unknown" but not None
        assert ctx.pipeline_version  # may be "unknown" but not None
        assert ctx.pipeline_commit  # may be "unknown" but not None
        assert ctx.pipeline_repo  # may be "unknown" but not None
        assert ctx.pipeline_description
        assert isinstance(ctx.pipeline_config, dict)
        assert isinstance(ctx.library_deps, list)


class TestDeterministicFields:
    """Tests for populate_deterministic_fields."""

    def _make_ctx(self, **overrides) -> MetadataContext:
        defaults = {
            "dd_version": "4.0.0",
            "access_layer_version": "5.2.0",
            "creation_date": "2025-01-01T00:00:00+00:00",
            "provider": "jet",
            "source": "imas-codex v4.1.0",
            "pipeline_version": "4.1.0",
            "pipeline_commit": "abc123",
            "pipeline_repo": "https://github.com/iterorganization/imas-codex",
            "pipeline_description": "IMAS mapping pipeline",
            "pipeline_config": {"test": True},
            "library_deps": [
                {
                    "name": "numpy",
                    "version": "1.26.0",
                    "repository": "https://github.com/numpy/numpy",
                    "description": "Numerical computing",
                    "commit": "unknown",
                }
            ],
        }
        defaults.update(overrides)
        return MetadataContext(**defaults)

    def test_all_deterministic_paths(self):
        """Verify all expected deterministic paths are present."""
        ctx = self._make_ctx()
        fields = populate_deterministic_fields(ctx, "pf_active")

        expected_paths = [
            "ids_properties/version_put/data_dictionary",
            "ids_properties/version_put/access_layer",
            "ids_properties/version_put/access_layer_language",
            "ids_properties/creation_date",
            "ids_properties/provider",
            "ids_properties/source",
            "code/name",
            "code/version",
            "code/repository",
            "code/commit",
            "code/description",
            "code/parameters",
        ]
        for path in expected_paths:
            assert path in fields, f"Missing path: {path}"

    def test_deterministic_values(self):
        """Verify deterministic field values match context."""
        ctx = self._make_ctx()
        fields = populate_deterministic_fields(ctx, "equilibrium")

        assert fields["ids_properties/version_put/data_dictionary"] == "4.0.0"
        assert fields["ids_properties/version_put/access_layer"] == "5.2.0"
        assert fields["ids_properties/version_put/access_layer_language"] == "python"
        assert fields["ids_properties/provider"] == "jet"
        assert fields["code/name"] == "imas-codex"
        assert fields["code/version"] == "4.1.0"
        assert fields["code/commit"] == "abc123"

    def test_library_entries(self):
        """Verify library entries are included as _library_entries."""
        ctx = self._make_ctx()
        fields = populate_deterministic_fields(ctx, "magnetics")

        assert "_library_entries" in fields
        assert len(fields["_library_entries"]) == 1
        assert fields["_library_entries"][0]["name"] == "numpy"

    def test_pipeline_config_json(self):
        """Verify pipeline_config is serialized to JSON."""
        ctx = self._make_ctx(pipeline_config={"model": "gemini", "batch_size": 10})
        fields = populate_deterministic_fields(ctx, "pf_active")

        parsed = json.loads(fields["code/parameters"])
        assert parsed["model"] == "gemini"
        assert parsed["batch_size"] == 10

    def test_empty_pipeline_config(self):
        """Verify empty pipeline_config serializes to empty JSON object."""
        ctx = self._make_ctx(pipeline_config={})
        fields = populate_deterministic_fields(ctx, "equilibrium")

        parsed = json.loads(fields["code/parameters"])
        assert parsed == {}

    def test_empty_library_deps(self):
        """Verify _library_entries is empty list when no deps."""
        ctx = self._make_ctx(library_deps=[])
        fields = populate_deterministic_fields(ctx, "pf_active")

        assert "_library_entries" in fields
        assert fields["_library_entries"] == []

    def test_source_value(self):
        """Verify source field is passed through from context."""
        ctx = self._make_ctx(source="imas-codex v99.0.0")
        fields = populate_deterministic_fields(ctx, "core_profiles")

        assert fields["ids_properties/source"] == "imas-codex v99.0.0"

    def test_creation_date_passthrough(self):
        """Verify creation_date is passed through from context."""
        ctx = self._make_ctx(creation_date="2025-06-15T12:00:00+00:00")
        fields = populate_deterministic_fields(ctx, "magnetics")

        assert fields["ids_properties/creation_date"] == "2025-06-15T12:00:00+00:00"


class TestMetadataPopulationResponse:
    """Tests for the Pydantic LLM response model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        resp = MetadataPopulationResponse(
            comment="JET pf_active mapping covering poloidal field coils",
            occurrence_type_name="experimental",
            occurrence_type_index=1,
            occurrence_type_description="Experimental data from JET",
            provenance_sources="JET MDSplus via PPF/JPF",
            homogeneous_time=1,
            homogeneous_time_reasoning="All signals share common time base",
        )
        assert resp.homogeneous_time == 1
        assert resp.occurrence_type_index == 1

    def test_invalid_homogeneous_time(self):
        """Test validation rejects invalid homogeneous_time."""
        with pytest.raises(ValueError, match="homogeneous_time must be 0, 1, or 2"):
            MetadataPopulationResponse(
                comment="test",
                homogeneous_time=5,
            )

    def test_invalid_occurrence_type_index(self):
        """Test validation rejects invalid occurrence_type_index."""
        with pytest.raises(ValueError, match="occurrence_type_index must be 0-3"):
            MetadataPopulationResponse(
                comment="test",
                occurrence_type_index=99,
            )

    def test_defaults(self):
        """Test default values are applied."""
        resp = MetadataPopulationResponse(comment="minimal test")
        assert resp.occurrence_type_name == "machine_description"
        assert resp.occurrence_type_index == 0
        assert resp.homogeneous_time == 1

    def test_homogeneous_time_zero(self):
        """Test homogeneous_time=0 (heterogeneous) is accepted."""
        resp = MetadataPopulationResponse(comment="test", homogeneous_time=0)
        assert resp.homogeneous_time == 0

    def test_homogeneous_time_two(self):
        """Test homogeneous_time=2 (independent) is accepted."""
        resp = MetadataPopulationResponse(comment="test", homogeneous_time=2)
        assert resp.homogeneous_time == 2

    def test_all_occurrence_type_indices(self):
        """Test all valid occurrence_type_index values (0-3) are accepted."""
        for idx in range(4):
            resp = MetadataPopulationResponse(comment="test", occurrence_type_index=idx)
            assert resp.occurrence_type_index == idx

    def test_provenance_sources_default(self):
        """Test provenance_sources defaults to empty string."""
        resp = MetadataPopulationResponse(comment="test")
        assert resp.provenance_sources == ""

    def test_occurrence_type_description_default(self):
        """Test occurrence_type_description defaults to empty string."""
        resp = MetadataPopulationResponse(comment="test")
        assert resp.occurrence_type_description == ""

    def test_homogeneous_time_reasoning_default(self):
        """Test homogeneous_time_reasoning defaults to empty string."""
        resp = MetadataPopulationResponse(comment="test")
        assert resp.homogeneous_time_reasoning == ""

    def test_negative_homogeneous_time_rejected(self):
        """Test negative homogeneous_time is rejected."""
        with pytest.raises(ValueError, match="homogeneous_time must be 0, 1, or 2"):
            MetadataPopulationResponse(comment="test", homogeneous_time=-1)

    def test_negative_occurrence_type_index_rejected(self):
        """Test negative occurrence_type_index is rejected."""
        with pytest.raises(ValueError, match="occurrence_type_index must be 0-3"):
            MetadataPopulationResponse(comment="test", occurrence_type_index=-1)


class TestPopulateMetadata:
    """Tests for the main populate_metadata function."""

    def test_populate_with_mocked_llm(self):
        """Test full populate_metadata with mocked LLM."""
        gc = MagicMock()
        gc.query.return_value = [{"doc": "Test IDS documentation"}]

        mock_response = MetadataPopulationResponse(
            comment="Test comment for magnetics",
            occurrence_type_name="experimental",
            occurrence_type_index=1,
            occurrence_type_description="Experimental magnetics data",
            provenance_sources="MDSplus database",
            homogeneous_time=1,
            homogeneous_time_reasoning="Common time base",
        )

        with (
            patch(
                "imas_codex.discovery.base.llm.call_llm_structured",
                return_value=(mock_response, 0.05, 500),
            ),
            patch("imas_codex.settings.get_model", return_value="test-model"),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="test prompt content",
            ),
        ):
            result = populate_metadata(
                "jet",
                "magnetics",
                gc=gc,
                dd_version="4.0.0",
                mapped_signals=[
                    {
                        "source_id": "jet:ip",
                        "target_id": "magnetics/ip/data",
                        "confidence": 0.95,
                    },
                ],
            )

        assert result.facility == "jet"
        assert result.ids_name == "magnetics"
        assert result.cost_usd == 0.05
        assert result.tokens == 500
        assert "ids_properties/comment" in result.llm_fields
        assert (
            result.llm_fields["ids_properties/comment"] == "Test comment for magnetics"
        )
        assert (
            "ids_properties/version_put/data_dictionary" in result.deterministic_fields
        )

    def test_populate_llm_failure_graceful(self):
        """Test that LLM failure returns result with empty llm_fields."""
        gc = MagicMock()
        gc.query.return_value = []

        with patch(
            "imas_codex.discovery.base.llm.call_llm_structured",
            side_effect=Exception("LLM timeout"),
        ):
            result = populate_metadata(
                "tcv",
                "equilibrium",
                gc=gc,
                dd_version="4.0.0",
            )

        assert result.llm_fields == {}
        assert result.cost_usd == 0.0
        assert len(result.deterministic_fields) > 0  # Deterministic still populated

    def test_populate_with_pipeline_config(self):
        """Test that pipeline_config is passed through."""
        gc = MagicMock()
        gc.query.return_value = []

        with patch(
            "imas_codex.discovery.base.llm.call_llm_structured",
            side_effect=Exception("skip LLM"),
        ):
            result = populate_metadata(
                "jet",
                "pf_active",
                gc=gc,
                dd_version="4.0.0",
                pipeline_config={"custom": "config"},
            )

        params = json.loads(result.deterministic_fields["code/parameters"])
        assert params["custom"] == "config"

    def test_populate_returns_ids_metadata_result(self):
        """Test that populate_metadata returns an IDSMetadataResult."""
        gc = MagicMock()
        gc.query.return_value = []

        result = populate_metadata(
            "jet",
            "pf_active",
            gc=gc,
            dd_version="4.0.0",
        )

        assert isinstance(result, IDSMetadataResult)
        assert result.facility == "jet"
        assert result.ids_name == "pf_active"
        assert result.dd_version == "4.0.0"

    def test_populate_deterministic_fields_always_present(self):
        """Test deterministic fields are populated even with LLM errors."""
        gc = MagicMock()
        gc.query.side_effect = Exception("graph down")

        result = populate_metadata(
            "mast-u",
            "magnetics",
            gc=gc,
            dd_version="4.0.0",
        )

        assert (
            "ids_properties/version_put/data_dictionary" in result.deterministic_fields
        )
        assert "ids_properties/provider" in result.deterministic_fields
        assert "code/name" in result.deterministic_fields
        assert result.deterministic_fields["ids_properties/provider"] == "mast-u"

    def test_populate_llm_fields_occurrence_type(self):
        """Test that occurrence_type fields are all present in llm_fields."""
        gc = MagicMock()
        gc.query.return_value = []

        mock_response = MetadataPopulationResponse(
            comment="Test",
            occurrence_type_name="experimental",
            occurrence_type_index=1,
            occurrence_type_description="Experimental data",
            homogeneous_time=0,
        )

        with (
            patch(
                "imas_codex.discovery.base.llm.call_llm_structured",
                return_value=(mock_response, 0.01, 100),
            ),
            patch("imas_codex.settings.get_model", return_value="test-model"),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="test prompt",
            ),
        ):
            result = populate_metadata(
                "tcv",
                "magnetics",
                gc=gc,
                dd_version="4.0.0",
            )

        assert "ids_properties/occurrence_type/name" in result.llm_fields
        assert "ids_properties/occurrence_type/index" in result.llm_fields
        assert "ids_properties/occurrence_type/description" in result.llm_fields
        assert "ids_properties/homogeneous_time" in result.llm_fields
        assert (
            result.llm_fields["ids_properties/occurrence_type/name"] == "experimental"
        )
        assert result.llm_fields["ids_properties/occurrence_type/index"] == 1
        assert result.llm_fields["ids_properties/homogeneous_time"] == 0

    def test_populate_provenance_included_when_nonempty(self):
        """Test provenance_sources is included in llm_fields when non-empty."""
        gc = MagicMock()
        gc.query.return_value = []

        mock_response = MetadataPopulationResponse(
            comment="Test",
            provenance_sources="TCV MDSplus via tcvpy",
            homogeneous_time=1,
        )

        with (
            patch(
                "imas_codex.discovery.base.llm.call_llm_structured",
                return_value=(mock_response, 0.01, 100),
            ),
            patch("imas_codex.settings.get_model", return_value="test-model"),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="test prompt",
            ),
        ):
            result = populate_metadata(
                "tcv",
                "equilibrium",
                gc=gc,
                dd_version="4.0.0",
            )

        assert "ids_properties/provenance/node/sources" in result.llm_fields
        assert (
            result.llm_fields["ids_properties/provenance/node/sources"]
            == "TCV MDSplus via tcvpy"
        )

    def test_populate_provenance_omitted_when_empty(self):
        """Test provenance_sources is omitted from llm_fields when empty."""
        gc = MagicMock()
        gc.query.return_value = []

        mock_response = MetadataPopulationResponse(
            comment="Test",
            provenance_sources="",
            homogeneous_time=1,
        )

        with (
            patch(
                "imas_codex.discovery.base.llm.call_llm_structured",
                return_value=(mock_response, 0.01, 100),
            ),
            patch("imas_codex.settings.get_model", return_value="test-model"),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="test prompt",
            ),
        ):
            result = populate_metadata(
                "tcv",
                "equilibrium",
                gc=gc,
                dd_version="4.0.0",
            )

        assert "ids_properties/provenance/node/sources" not in result.llm_fields

    def test_populate_no_signals_produces_fallback_description(self):
        """Test that no signals arg still succeeds (uses fallback message)."""
        gc = MagicMock()
        gc.query.return_value = []

        result = populate_metadata(
            "jet",
            "pf_active",
            gc=gc,
            dd_version="4.0.0",
            mapped_signals=None,
        )

        # Should succeed and return a result regardless
        assert isinstance(result, IDSMetadataResult)

    def test_populate_dd_version_in_result(self):
        """Test that dd_version from input appears in the result."""
        gc = MagicMock()
        gc.query.return_value = []

        result = populate_metadata(
            "jet",
            "pf_active",
            gc=gc,
            dd_version="3.39.0",
        )

        assert result.dd_version == "3.39.0"


class TestPersistMetadata:
    """Tests for persist_metadata."""

    def test_persist_writes_to_graph(self):
        """Test that persist_metadata writes JSON to IMASMapping node."""
        gc = MagicMock()

        result = IDSMetadataResult(
            facility="jet",
            ids_name="pf_active",
            dd_version="4.0.0",
            deterministic_fields={
                "ids_properties/provider": "jet",
                "code/name": "imas-codex",
                "_library_entries": [{"name": "numpy", "version": "1.26.0"}],
            },
            llm_fields={
                "ids_properties/comment": "JET pf_active test",
            },
        )

        persist_metadata(result, "jet:pf_active", gc=gc)

        gc.query.assert_called_once()
        call_kwargs = gc.query.call_args
        # Verify the Cypher query mentions IMASMapping
        assert "IMASMapping" in call_kwargs.args[0]
        # Verify parameters
        assert call_kwargs.kwargs["mapping_id"] == "jet:pf_active"
        # Verify JSON properties are passed
        ids_props = json.loads(call_kwargs.kwargs["ids_props"])
        assert "ids_properties/provider" in ids_props
        assert "ids_properties/comment" in ids_props
        code_meta = json.loads(call_kwargs.kwargs["code_meta"])
        assert "code/name" in code_meta

    def test_persist_excludes_internal_keys(self):
        """Test that _library_entries is not included in ids_props or code_meta."""
        gc = MagicMock()

        result = IDSMetadataResult(
            facility="jet",
            ids_name="pf_active",
            dd_version="4.0.0",
            deterministic_fields={
                "ids_properties/provider": "jet",
                "code/name": "imas-codex",
                "_library_entries": [{"name": "numpy", "version": "1.26.0"}],
            },
            llm_fields={},
        )

        persist_metadata(result, "jet:pf_active", gc=gc)

        call_kwargs = gc.query.call_args
        ids_props = json.loads(call_kwargs.kwargs["ids_props"])
        code_meta = json.loads(call_kwargs.kwargs["code_meta"])

        assert "_library_entries" not in ids_props
        assert "_library_entries" not in code_meta

    def test_persist_library_entries_stored_separately(self):
        """Test that _library_entries are stored in library_meta."""
        gc = MagicMock()

        result = IDSMetadataResult(
            facility="jet",
            ids_name="pf_active",
            dd_version="4.0.0",
            deterministic_fields={
                "ids_properties/provider": "jet",
                "_library_entries": [
                    {"name": "numpy", "version": "1.26.0"},
                    {"name": "pydantic", "version": "2.5.0"},
                ],
            },
            llm_fields={},
        )

        persist_metadata(result, "jet:pf_active", gc=gc)

        call_kwargs = gc.query.call_args
        library_meta = json.loads(call_kwargs.kwargs["library_meta"])
        assert len(library_meta) == 2
        assert library_meta[0]["name"] == "numpy"

    def test_persist_merges_deterministic_and_llm_fields(self):
        """Test that both deterministic and llm fields appear in the stored data."""
        gc = MagicMock()

        result = IDSMetadataResult(
            facility="tcv",
            ids_name="equilibrium",
            dd_version="4.0.0",
            deterministic_fields={
                "ids_properties/provider": "tcv",
                "ids_properties/version_put/data_dictionary": "4.0.0",
            },
            llm_fields={
                "ids_properties/comment": "TCV equilibrium data",
                "ids_properties/homogeneous_time": 1,
            },
        )

        persist_metadata(result, "tcv:equilibrium", gc=gc)

        call_kwargs = gc.query.call_args
        ids_props = json.loads(call_kwargs.kwargs["ids_props"])
        # Both deterministic and LLM ids_properties fields should be present
        assert "ids_properties/provider" in ids_props
        assert "ids_properties/comment" in ids_props
        assert "ids_properties/version_put/data_dictionary" in ids_props


class TestFormatSignalsSummary:
    """Tests for _format_signals_summary helper."""

    def test_empty_signals(self):
        """Test None and empty list both return the fallback message."""
        assert _format_signals_summary(None) == "No signal mappings available."
        assert _format_signals_summary([]) == "No signal mappings available."

    def test_format_basic(self):
        """Test basic signal formatting includes source, target, and confidence."""
        signals = [
            {
                "source_id": "jet:ip",
                "target_id": "magnetics/ip/data",
                "confidence": 0.95,
            },
        ]
        result = _format_signals_summary(signals)
        assert "jet:ip" in result
        assert "magnetics/ip/data" in result
        assert "0.95" in result

    def test_truncation(self):
        """Test that more than 50 signals produces truncation notice."""
        signals = [
            {"source_id": f"sig_{i}", "target_id": f"path_{i}", "confidence": 0.5}
            for i in range(60)
        ]
        result = _format_signals_summary(signals)
        assert "... and 10 more" in result

    def test_no_truncation_at_50(self):
        """Test that exactly 50 signals produces no truncation notice."""
        signals = [
            {"source_id": f"sig_{i}", "target_id": f"path_{i}", "confidence": 0.5}
            for i in range(50)
        ]
        result = _format_signals_summary(signals)
        assert "more" not in result

    def test_arrow_format(self):
        """Test the arrow format between source and target."""
        signals = [
            {"source_id": "src", "target_id": "tgt", "confidence": 1.0},
        ]
        result = _format_signals_summary(signals)
        assert "src → tgt" in result

    def test_missing_keys_fallback(self):
        """Test graceful handling of signals with missing keys."""
        signals = [{}]
        result = _format_signals_summary(signals)
        assert "unknown" in result

    def test_multiple_signals_multiline(self):
        """Test that multiple signals produce multiple lines."""
        signals = [
            {"source_id": f"sig_{i}", "target_id": f"path_{i}", "confidence": 0.8}
            for i in range(3)
        ]
        result = _format_signals_summary(signals)
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 3
