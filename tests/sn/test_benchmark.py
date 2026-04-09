"""Tests for the SN benchmarking system."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from imas_standard_names.grammar import compose_standard_name, parse_standard_name

# -----------------------------------------------------------------------
# Reference dataset tests
# -----------------------------------------------------------------------


class TestReferenceDataset:
    """Verify the reference dataset is valid and self-consistent."""

    def test_reference_not_empty(self):
        from imas_codex.sn.benchmark_reference import REFERENCE_NAMES

        assert len(REFERENCE_NAMES) >= 20, "Reference set should have >= 20 entries"

    def test_all_names_round_trip(self):
        """Every reference name must survive parse→compose round-trip."""
        from imas_codex.sn.benchmark_reference import REFERENCE_NAMES

        failures = []
        for path, entry in REFERENCE_NAMES.items():
            name = entry["name"]
            try:
                parsed = parse_standard_name(name)
                rt = compose_standard_name(parsed)
                if rt != name:
                    failures.append(f"{path}: {name!r} → {rt!r}")
            except Exception as e:
                failures.append(f"{path}: {name!r} raised {e!s:.80s}")

        assert not failures, "Round-trip failures:\n" + "\n".join(failures)

    def test_all_entries_have_required_keys(self):
        from imas_codex.sn.benchmark_reference import REFERENCE_NAMES

        for path, entry in REFERENCE_NAMES.items():
            assert "name" in entry, f"Missing 'name' in {path}"
            assert "fields" in entry, f"Missing 'fields' in {path}"
            assert isinstance(entry["name"], str), f"name must be str in {path}"
            assert isinstance(entry["fields"], dict), f"fields must be dict in {path}"

    def test_all_fields_have_physical_or_geometric_base(self):
        from imas_codex.sn.benchmark_reference import REFERENCE_NAMES

        for path, entry in REFERENCE_NAMES.items():
            fields = entry["fields"]
            has_physical = "physical_base" in fields
            has_geometric = "geometric_base" in fields
            assert has_physical or has_geometric, (
                f"{path}: must have physical_base or geometric_base, got {fields}"
            )

    def test_paths_look_like_dd_paths(self):
        from imas_codex.sn.benchmark_reference import REFERENCE_NAMES

        for path in REFERENCE_NAMES:
            assert "/" in path, f"Path should contain '/': {path}"
            assert not path.startswith("/"), f"Path should not start with '/': {path}"


# -----------------------------------------------------------------------
# Dataclass tests
# -----------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass instantiation and basic behavior."""

    def test_benchmark_config_defaults(self):
        from imas_codex.sn.benchmark import BenchmarkConfig

        cfg = BenchmarkConfig(models=["model-a", "model-b"])
        assert cfg.source == "dd"
        assert cfg.max_candidates == 50
        assert cfg.runs_per_model == 1
        assert cfg.temperature == 0.0
        assert cfg.ids_filter is None

    def test_benchmark_config_custom(self):
        from imas_codex.sn.benchmark import BenchmarkConfig

        cfg = BenchmarkConfig(
            models=["m1"],
            source="signals",
            ids_filter="equilibrium",
            domain_filter="magnetics",
            facility="tcv",
            max_candidates=100,
            runs_per_model=3,
            temperature=0.5,
        )
        assert cfg.facility == "tcv"
        assert cfg.runs_per_model == 3

    def test_model_result_defaults(self):
        from imas_codex.sn.benchmark import ModelResult

        r = ModelResult(model="test-model")
        assert r.model == "test-model"
        assert r.candidates == []
        assert r.grammar_valid_count == 0
        assert r.total_cost == 0.0
        assert r.names_per_minute == 0.0

    def test_benchmark_report_instantiation(self):
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
        )

        cfg = BenchmarkConfig(models=["m"])
        mr = ModelResult(model="m", grammar_valid_count=5)
        report = BenchmarkReport(
            config=cfg,
            results=[mr],
            reference_names=["a", "b"],
            extraction_count=10,
            timestamp="2025-01-01T00:00:00",
        )
        assert report.extraction_count == 10
        assert len(report.results) == 1


# -----------------------------------------------------------------------
# Grammar context builder tests
# -----------------------------------------------------------------------


class TestGrammarContext:
    """Verify grammar context builder provides all template variables."""

    def test_build_grammar_context_keys(self):
        from imas_codex.sn.benchmark import build_grammar_context

        ctx = build_grammar_context()
        expected_keys = {
            "subjects",
            "positions",
            "components",
            "coordinates",
            "processes",
            "transformations",
            "geometric_bases",
            "objects",
            "binary_operators",
        }
        assert set(ctx.keys()) == expected_keys

    def test_all_values_non_empty(self):
        from imas_codex.sn.benchmark import build_grammar_context

        ctx = build_grammar_context()
        for key, values in ctx.items():
            assert len(values) > 0, f"{key} should have at least one value"
            assert all(isinstance(v, str) for v in values), (
                f"{key} values must be strings"
            )


# -----------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------


class TestValidation:
    """Test the candidate validation logic."""

    def test_valid_candidate(self):
        from imas_codex.sn.benchmark import validate_candidate

        candidate = {
            "standard_name": "electron_temperature",
            "fields": {"physical_base": "temperature", "subject": "electron"},
        }
        g_valid, f_consistent = validate_candidate(candidate)
        assert g_valid is True
        assert f_consistent is True

    def test_invalid_grammar(self):
        from imas_codex.sn.benchmark import validate_candidate

        candidate = {
            "standard_name": "this_is_not_valid_!!!",
            "fields": {"physical_base": "nonsense"},
        }
        g_valid, f_consistent = validate_candidate(candidate)
        assert g_valid is False
        assert f_consistent is False

    def test_valid_grammar_inconsistent_fields(self):
        from imas_codex.sn.benchmark import validate_candidate

        candidate = {
            "standard_name": "electron_temperature",
            "fields": {"physical_base": "density", "subject": "ion"},
        }
        g_valid, f_consistent = validate_candidate(candidate)
        assert g_valid is True
        assert f_consistent is False

    def test_empty_candidate(self):
        from imas_codex.sn.benchmark import validate_candidate

        g_valid, f_consistent = validate_candidate({})
        assert g_valid is False
        assert f_consistent is False


# -----------------------------------------------------------------------
# Reference comparison tests
# -----------------------------------------------------------------------


class TestReferenceComparison:
    """Test reference set comparison logic."""

    def test_full_overlap(self):
        from imas_codex.sn.benchmark import compare_to_reference

        reference = {
            "path/a": {"name": "electron_temperature", "fields": {}},
            "path/b": {"name": "safety_factor", "fields": {}},
        }
        candidates = [
            {"source_id": "path/a", "standard_name": "electron_temperature"},
            {"source_id": "path/b", "standard_name": "safety_factor"},
        ]
        overlap, total, precision, recall = compare_to_reference(candidates, reference)
        assert overlap == 2
        assert total == 2
        assert recall == 1.0

    def test_no_overlap(self):
        from imas_codex.sn.benchmark import compare_to_reference

        reference = {
            "path/a": {"name": "electron_temperature", "fields": {}},
        }
        candidates = [
            {"source_id": "path/a", "standard_name": "ion_temperature"},
        ]
        overlap, total, precision, recall = compare_to_reference(candidates, reference)
        assert overlap == 0
        assert total == 1
        assert recall == 0.0

    def test_partial_overlap(self):
        from imas_codex.sn.benchmark import compare_to_reference

        reference = {
            "path/a": {"name": "electron_temperature", "fields": {}},
            "path/b": {"name": "safety_factor", "fields": {}},
        }
        candidates = [
            {"source_id": "path/a", "standard_name": "electron_temperature"},
            {"source_id": "path/b", "standard_name": "beta"},
            {"source_id": "path/c", "standard_name": "elongation"},
        ]
        overlap, total, precision, recall = compare_to_reference(candidates, reference)
        assert overlap == 1
        assert total == 2
        assert recall == 0.5

    def test_empty_candidates(self):
        from imas_codex.sn.benchmark import compare_to_reference

        overlap, total, precision, recall = compare_to_reference(
            [], {"path/a": {"name": "x", "fields": {}}}
        )
        assert overlap == 0
        assert precision == 0.0


# -----------------------------------------------------------------------
# JSON serialization tests
# -----------------------------------------------------------------------


class TestJsonSerialization:
    """Test JSON round-trip for BenchmarkReport."""

    def test_json_round_trip(self):
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
        )

        cfg = BenchmarkConfig(
            models=["model-a", "model-b"],
            source="dd",
            ids_filter="equilibrium",
            max_candidates=25,
        )
        r1 = ModelResult(
            model="model-a",
            candidates=[{"source_id": "x", "standard_name": "electron_temperature"}],
            grammar_valid_count=1,
            grammar_invalid_count=0,
            total_cost=0.05,
            total_tokens=500,
            elapsed_seconds=12.5,
            names_per_minute=4.8,
            cost_per_name=0.05,
        )
        r2 = ModelResult(model="model-b")
        report = BenchmarkReport(
            config=cfg,
            results=[r1, r2],
            reference_names=["path/a", "path/b"],
            extraction_count=10,
            timestamp="2025-01-15T12:00:00+00:00",
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        # Verify structure
        assert parsed["config"]["models"] == ["model-a", "model-b"]
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["model"] == "model-a"
        assert parsed["results"][0]["total_cost"] == 0.05
        assert parsed["extraction_count"] == 10

    def test_from_json(self):
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
        )

        cfg = BenchmarkConfig(models=["m1"])
        r = ModelResult(model="m1", grammar_valid_count=3)
        original = BenchmarkReport(
            config=cfg,
            results=[r],
            reference_names=["a"],
            extraction_count=5,
            timestamp="2025-01-01",
        )

        json_str = original.to_json()
        restored = BenchmarkReport.from_json(json_str)

        assert restored.config.models == ["m1"]
        assert restored.results[0].model == "m1"
        assert restored.results[0].grammar_valid_count == 3
        assert restored.extraction_count == 5
        assert restored.timestamp == "2025-01-01"


# -----------------------------------------------------------------------
# Rich table rendering tests
# -----------------------------------------------------------------------


class TestRichTable:
    """Verify the Rich comparison table renders without error."""

    def test_render_empty_report(self):
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            render_comparison_table,
        )

        report = BenchmarkReport(
            config=BenchmarkConfig(models=[]),
            results=[],
            reference_names=[],
            timestamp="2025-01-01",
        )
        # Should not raise
        render_comparison_table(report)

    def test_render_with_results(self):
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
            render_comparison_table,
        )

        r = ModelResult(
            model="test-model",
            candidates=[{"source_id": "p", "standard_name": "electron_temperature"}],
            grammar_valid_count=1,
            grammar_invalid_count=0,
            fields_consistent_count=1,
            total_cost=0.01,
            total_tokens=100,
            elapsed_seconds=5.0,
            names_per_minute=12.0,
            cost_per_name=0.01,
            reference_overlap=1,
            reference_total=10,
        )
        report = BenchmarkReport(
            config=BenchmarkConfig(models=["test-model"]),
            results=[r],
            reference_names=["a"],
            extraction_count=5,
            timestamp="2025-01-01",
        )
        # Should not raise
        render_comparison_table(report)

    def test_render_with_zero_candidates(self):
        """Model that produced zero candidates should show '—' not crash."""
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
            render_comparison_table,
        )

        r = ModelResult(model="empty-model")
        report = BenchmarkReport(
            config=BenchmarkConfig(models=["empty-model"]),
            results=[r],
            reference_names=[],
            timestamp="2025-01-01",
        )
        render_comparison_table(report)


# -----------------------------------------------------------------------
# Benchmark runner tests (mocked LLM)
# -----------------------------------------------------------------------


class TestBenchmarkRunner:
    """Test the async benchmark runner with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_run_benchmark_mocked(self):
        """Run benchmark with mocked extraction and LLM calls."""
        from imas_codex.sn.benchmark import BenchmarkConfig, run_benchmark
        from imas_codex.sn.models import SNCandidate, SNComposeBatch

        config = BenchmarkConfig(
            models=["mock-model-a"],
            max_candidates=5,
        )

        # Fake extraction batches
        fake_batches = [
            {
                "group_key": "equilibrium",
                "items": [
                    {
                        "path": "equilibrium/time_slice/profiles_1d/safety_factor",
                        "description": "Safety factor",
                        "units": None,
                        "data_type": "FLT_1D",
                        "cluster_label": "safety_factor",
                    },
                    {
                        "path": "core_profiles/profiles_1d/electrons/temperature",
                        "description": "Electron temperature",
                        "units": "eV",
                        "data_type": "FLT_1D",
                        "cluster_label": "electron_temperature",
                    },
                ],
                "existing_names": [],
            }
        ]

        # Mock LLM response
        mock_response = SNComposeBatch(
            candidates=[
                SNCandidate(
                    source_id="equilibrium/time_slice/profiles_1d/safety_factor",
                    standard_name="safety_factor",
                    fields={"physical_base": "safety_factor"},
                    confidence=0.95,
                    reason="Safety factor profile",
                ),
                SNCandidate(
                    source_id="core_profiles/profiles_1d/electrons/temperature",
                    standard_name="electron_temperature",
                    fields={"physical_base": "temperature", "subject": "electron"},
                    confidence=0.98,
                    reason="Electron temperature",
                ),
            ],
            skipped=[],
        )

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                return_value=(mock_response, 0.01, 200),
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="mocked prompt text",
            ),
        ):
            report = await run_benchmark(config, extraction_batches=fake_batches)

        assert len(report.results) == 1
        r = report.results[0]
        assert r.model == "mock-model-a"
        assert len(r.candidates) == 2
        assert r.grammar_valid_count == 2
        assert r.grammar_invalid_count == 0
        assert r.total_cost == 0.01
        assert r.total_tokens == 200
        assert r.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_benchmark_multiple_models(self):
        """Run with multiple models and verify separate results."""
        from imas_codex.sn.benchmark import BenchmarkConfig, run_benchmark
        from imas_codex.sn.models import SNCandidate, SNComposeBatch

        config = BenchmarkConfig(models=["model-a", "model-b"], max_candidates=2)

        fake_batches = [
            {
                "group_key": "test",
                "items": [
                    {
                        "path": "equilibrium/time_slice/profiles_1d/elongation",
                        "description": "Elongation",
                        "units": None,
                        "data_type": "FLT_1D",
                        "cluster_label": None,
                    },
                ],
                "existing_names": [],
            }
        ]

        mock_response = SNComposeBatch(
            candidates=[
                SNCandidate(
                    source_id="equilibrium/time_slice/profiles_1d/elongation",
                    standard_name="elongation",
                    fields={"physical_base": "elongation"},
                    confidence=0.9,
                    reason="Plasma elongation",
                ),
            ],
            skipped=[],
        )

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                return_value=(mock_response, 0.005, 100),
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="prompt",
            ),
        ):
            report = await run_benchmark(config, extraction_batches=fake_batches)

        assert len(report.results) == 2
        assert report.results[0].model == "model-a"
        assert report.results[1].model == "model-b"
        # Both should have the same structure since same mock
        for r in report.results:
            assert len(r.candidates) == 1
            assert r.grammar_valid_count == 1

    @pytest.mark.asyncio
    async def test_run_benchmark_llm_failure(self):
        """LLM call failure should be recorded, not crash."""
        from imas_codex.sn.benchmark import BenchmarkConfig, run_benchmark

        config = BenchmarkConfig(models=["fail-model"], max_candidates=2)

        fake_batches = [
            {
                "group_key": "test",
                "items": [
                    {
                        "path": "test/path",
                        "description": "Test",
                        "units": None,
                        "data_type": "FLT_0D",
                        "cluster_label": None,
                    }
                ],
                "existing_names": [],
            }
        ]

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM unavailable"),
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="prompt",
            ),
        ):
            report = await run_benchmark(config, extraction_batches=fake_batches)

        assert len(report.results) == 1
        r = report.results[0]
        assert r.batch_errors == 1
        assert len(r.candidates) == 0


# -----------------------------------------------------------------------
# CLI command tests
# -----------------------------------------------------------------------


class TestCLICommand:
    """Verify the CLI benchmark command is registered and callable."""

    def test_command_exists(self):
        from imas_codex.cli.sn import sn

        cmd = sn.get_command(None, "benchmark")
        assert cmd is not None, "benchmark command should be registered"

    def test_command_help(self):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        runner = CliRunner()
        result = runner.invoke(sn, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.output
        assert "--max-candidates" in result.output
        assert "--output" in result.output
        assert "--temperature" in result.output

    def test_command_requires_models(self):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        runner = CliRunner()
        result = runner.invoke(sn, ["benchmark"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()
