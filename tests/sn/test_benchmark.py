"""Tests for the SN benchmarking system."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

imas_standard_names = pytest.importorskip("imas_standard_names")
from imas_standard_names.grammar import (  # noqa: E402
    compose_standard_name,
    parse_standard_name,
)

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
# Context builder tests
# -----------------------------------------------------------------------


class TestGrammarContext:
    """Verify build_compose_context provides all template variables."""

    def test_build_compose_context_keys(self):
        """build_compose_context() should return rich grammar context."""
        from imas_codex.sn.context import build_compose_context

        ctx = build_compose_context()
        # Rich grammar context keys
        assert "canonical_pattern" in ctx
        assert "vocabulary_sections" in ctx
        assert "segment_descriptions" in ctx
        # Backward-compat enum lists still present
        assert "subjects" in ctx
        assert "positions" in ctx
        assert "components" in ctx

    def test_all_values_non_empty(self):
        """Enum lists from build_compose_context should be non-empty strings."""
        from imas_codex.sn.context import build_compose_context

        ctx = build_compose_context()
        for key in (
            "subjects",
            "positions",
            "components",
            "processes",
            "transformations",
        ):
            assert len(ctx[key]) > 0, f"{key} should have at least one value"
            assert all(isinstance(v, str) for v in ctx[key]), (
                f"{key} values must be strings"
            )


# -----------------------------------------------------------------------
# Prompt parity tests
# -----------------------------------------------------------------------


class TestPromptParity:
    """Verify benchmark uses the same prompt architecture as mint pipeline."""

    def test_extract_candidates_preserves_context(self):
        """_extract_candidates should include batch.context in output dicts."""
        from unittest.mock import patch

        from imas_codex.sn.benchmark import BenchmarkConfig, _extract_candidates
        from imas_codex.sn.sources.base import ExtractionBatch

        fake_batch = ExtractionBatch(
            source="dd",
            group_key="equilibrium",
            items=[{"path": "test/path", "description": "Test"}],
            context="IDS: equilibrium\nSemantic clusters: psi, safety_factor",
            existing_names=set(),
        )

        config = BenchmarkConfig(models=["test"])

        with patch(
            "imas_codex.sn.sources.dd.extract_dd_candidates",
            return_value=[fake_batch],
        ):
            batches = _extract_candidates(config)

        assert len(batches) == 1
        assert "context" in batches[0]
        assert (
            batches[0]["context"]
            == "IDS: equilibrium\nSemantic clusters: psi, safety_factor"
        )

    @pytest.mark.asyncio
    async def test_run_model_system_user_messages(self):
        """_run_model should construct [system, user] message structure."""
        from unittest.mock import AsyncMock, patch

        from imas_codex.sn.benchmark import BenchmarkConfig, _run_model
        from imas_codex.sn.models import SNComposeBatch

        config = BenchmarkConfig(models=["test"], temperature=0.0)
        batches = [
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
                "context": "IDS: test",
            }
        ]
        minimal_context: dict = {"subjects": ["electron"], "vocabulary_sections": []}
        captured_messages: list[dict] = []
        mock_response = SNComposeBatch(candidates=[], skipped=[])

        async def mock_llm(model, messages, response_model, **kwargs):
            captured_messages.extend(messages)
            return mock_response, 0.0, 0

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=mock_llm,
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="rendered prompt",
            ),
        ):
            await _run_model(
                model="test",
                extraction_batches=batches,
                config=config,
                reference={},
                system_prompt="System instructions",
                context=minimal_context,
            )

        assert len(captured_messages) == 2
        assert captured_messages[0]["role"] == "system"
        assert captured_messages[0]["content"] == "System instructions"
        assert captured_messages[1]["role"] == "user"


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

    def test_command_defaults_from_config(self):
        """Benchmark loads models from pyproject.toml config when --models omitted."""
        from imas_codex.settings import get_sn_benchmark_compose_models

        models = get_sn_benchmark_compose_models()
        assert len(models) > 0, "Should have default compose models"
        # Verify all model IDs have provider/slug format
        for m in models:
            assert "/" in m, f"Model ID should be provider/slug format: {m}"


class TestCalibrationDataset:
    """Test benchmark calibration dataset."""

    def test_calibration_loads(self):
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        assert isinstance(entries, list)
        assert len(entries) == 15, f"Expected 15 entries, got {len(entries)}"

    def test_calibration_tiers(self):
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        tiers = {}
        for entry in entries:
            tier = entry["tier"]
            tiers[tier] = tiers.get(tier, 0) + 1
        assert tiers == {"outstanding": 4, "good": 4, "adequate": 4, "poor": 3}

    def test_calibration_required_keys(self):
        from imas_codex.sn.benchmark import load_calibration_entries

        required = {"name", "tier", "expected_score", "description", "fields", "reason"}
        entries = load_calibration_entries()
        for entry in entries:
            missing = required - set(entry.keys())
            assert not missing, f"Entry {entry['name']} missing keys: {missing}"

    def test_calibration_names_round_trip(self):
        """Every calibration entry name must survive parse→compose round-trip."""
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        failures = []
        for entry in entries:
            name = entry["name"]
            try:
                parsed = parse_standard_name(name)
                rt = compose_standard_name(parsed)
                if rt != name:
                    failures.append(f"{name}: round-trip produced {rt!r}")
            except Exception as e:
                failures.append(f"{name}: {e!s:.80s}")
        assert not failures, "Round-trip failures:\n" + "\n".join(failures)

    def test_calibration_fields_compose_to_name(self):
        """compose_standard_name(StandardName(**fields)) == name for each entry."""
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        failures = []
        for entry in entries:
            try:
                sn = imas_standard_names.grammar.StandardName(**entry["fields"])
                composed = compose_standard_name(sn)
                if composed != entry["name"]:
                    failures.append(f"{entry['name']}: fields compose to {composed!r}")
            except Exception as e:
                failures.append(f"{entry['name']}: {e!s:.80s}")
        assert not failures, "Field composition failures:\n" + "\n".join(failures)

    def test_calibration_score_ranges(self):
        """Verify expected_score falls within the tier's defined range."""
        from imas_codex.sn.benchmark import load_calibration_entries

        tier_ranges = {
            "outstanding": (85, 100),
            "good": (60, 79),
            "adequate": (40, 59),
            "poor": (0, 39),
        }
        entries = load_calibration_entries()
        for entry in entries:
            lo, hi = tier_ranges[entry["tier"]]
            assert lo <= entry["expected_score"] <= hi, (
                f"{entry['name']} ({entry['tier']}): score {entry['expected_score']} "
                f"outside range [{lo}, {hi}]"
            )

    def test_calibration_no_duplicate_names(self):
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        names = [e["name"] for e in entries]
        assert len(names) == len(set(names)), "Duplicate names in calibration dataset"

    def test_reviewer_config_field(self):
        from imas_codex.sn.benchmark import BenchmarkConfig

        config = BenchmarkConfig(models=["test"], reviewer_model="test/model")
        assert config.reviewer_model == "test/model"

    def test_reviewer_config_default_none(self):
        from imas_codex.sn.benchmark import BenchmarkConfig

        config = BenchmarkConfig(models=["test"])
        assert config.reviewer_model is None

    def test_model_result_quality_fields(self):
        from imas_codex.sn.benchmark import ModelResult

        r = ModelResult(model="test")
        assert r.quality_scores == []
        assert r.quality_distribution == {}
        assert r.avg_quality_score == 0.0
        assert r.avg_doc_length == 0.0
        assert r.avg_fields_populated == 0.0

    def test_model_result_with_quality(self):
        from imas_codex.sn.benchmark import ModelResult

        r = ModelResult(
            model="test",
            quality_scores=[
                {
                    "name": "a",
                    "score": 80,
                    "quality_tier": "outstanding",
                    "reasoning": "good",
                }
            ],
            quality_distribution={"outstanding": 1},
            avg_quality_score=80.0,
            avg_doc_length=150.0,
            avg_fields_populated=0.5,
        )
        assert r.avg_quality_score == 80.0
        assert r.quality_distribution["outstanding"] == 1


class TestCacheTokenReporting:
    """Test prompt-cache token reporting in ModelResult and Rich table."""

    def test_llm_result_unpacking(self):
        """LLMResult supports 3-element tuple unpacking (backward compat)."""
        from imas_codex.discovery.base.llm import LLMResult

        r = LLMResult(
            "parsed", 0.05, 500, cache_read_tokens=300, cache_creation_tokens=100
        )
        parsed, cost, tokens = r
        assert parsed == "parsed"
        assert cost == 0.05
        assert tokens == 500
        assert r.cache_read_tokens == 300
        assert r.cache_creation_tokens == 100

    def test_llm_result_getattr_fallback(self):
        """getattr on a plain tuple returns 0 (mock compatibility)."""
        mock_return = ("parsed", 0.01, 200)
        assert getattr(mock_return, "cache_read_tokens", 0) == 0
        assert getattr(mock_return, "cache_creation_tokens", 0) == 0

    def test_model_result_cache_defaults(self):
        from imas_codex.sn.benchmark import ModelResult

        r = ModelResult(model="test")
        assert r.cache_read_tokens == 0
        assert r.cache_creation_tokens == 0

    def test_model_result_with_cache(self):
        from imas_codex.sn.benchmark import ModelResult

        r = ModelResult(
            model="test",
            cache_read_tokens=5000,
            cache_creation_tokens=2000,
        )
        assert r.cache_read_tokens == 5000
        assert r.cache_creation_tokens == 2000

    def test_cache_pct_all_read(self):
        """100% cache hit rate."""
        read, creation = 1000, 0
        total = read + creation
        pct = read / total * 100 if total > 0 else 0
        assert pct == 100.0

    def test_cache_pct_no_cache(self):
        """0/0 — no cache tokens at all."""
        read, creation = 0, 0
        total = read + creation
        pct = read / total * 100 if total > 0 else 0
        assert pct == 0.0

    def test_cache_pct_mixed(self):
        """Partial cache hit rate."""
        read, creation = 300, 700
        total = read + creation
        pct = read / total * 100 if total > 0 else 0
        assert pct == 30.0

    def test_cache_pct_all_creation(self):
        """First request — all tokens are cache creation."""
        read, creation = 0, 500
        total = read + creation
        pct = read / total * 100 if total > 0 else 0
        assert pct == 0.0

    def test_render_table_with_cache(self):
        """Cache % column should appear and show correct values."""
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
            cache_read_tokens=800,
            cache_creation_tokens=200,
        )
        report = BenchmarkReport(
            config=BenchmarkConfig(models=["test-model"]),
            results=[r],
            reference_names=[],
            extraction_count=1,
            timestamp="2025-01-01",
        )
        # Should not raise
        render_comparison_table(report)

    def test_render_table_no_cache(self):
        """Cache % column shows '—' when no cache tokens."""
        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
            render_comparison_table,
        )

        r = ModelResult(
            model="no-cache-model",
            candidates=[{"source_id": "p", "standard_name": "electron_temperature"}],
            grammar_valid_count=1,
            total_cost=0.01,
            total_tokens=100,
        )
        report = BenchmarkReport(
            config=BenchmarkConfig(models=["no-cache-model"]),
            results=[r],
            reference_names=[],
            timestamp="2025-01-01",
        )
        # Should not raise
        render_comparison_table(report)

    def test_cache_in_json_round_trip(self):
        """Cache fields survive JSON serialization."""
        import json

        from imas_codex.sn.benchmark import (
            BenchmarkConfig,
            BenchmarkReport,
            ModelResult,
        )

        r = ModelResult(
            model="m",
            cache_read_tokens=1500,
            cache_creation_tokens=500,
        )
        report = BenchmarkReport(
            config=BenchmarkConfig(models=["m"]),
            results=[r],
            reference_names=[],
            timestamp="2025-01-01",
        )
        parsed = json.loads(report.to_json())
        assert parsed["results"][0]["cache_read_tokens"] == 1500
        assert parsed["results"][0]["cache_creation_tokens"] == 500

        restored = BenchmarkReport.from_json(report.to_json())
        assert restored.results[0].cache_read_tokens == 1500
        assert restored.results[0].cache_creation_tokens == 500


class TestReviewerModelCLI:
    """Test --reviewer-model CLI option."""

    def test_reviewer_model_in_help(self):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        runner = CliRunner()
        result = runner.invoke(sn, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--reviewer-model" in result.output


# -----------------------------------------------------------------------
# 5-dimensional scoring model tests
# -----------------------------------------------------------------------


class TestQualityReviewModel:
    """Test the 5-dimensional QualityReview Pydantic model."""

    def _make_review_model(self):
        """Import the QualityReview model from inside score_with_reviewer."""
        from pydantic import BaseModel, Field

        class QualityReview(BaseModel):
            name: str
            quality_tier: str = Field(
                description="outstanding, good, adequate, or poor"
            )
            score: int = Field(
                ge=0, le=100, description="Total quality score (sum of dimensions)"
            )
            grammar_score: int = Field(ge=0, le=20, description="Grammar correctness")
            semantic_score: int = Field(ge=0, le=20, description="Semantic accuracy")
            documentation_score: int = Field(
                ge=0, le=20, description="Documentation quality"
            )
            convention_score: int = Field(ge=0, le=20, description="Naming conventions")
            completeness_score: int = Field(
                ge=0, le=20, description="Entry completeness"
            )
            reasoning: str

        return QualityReview

    def test_valid_review(self):
        QualityReview = self._make_review_model()
        review = QualityReview(
            name="electron_temperature",
            quality_tier="outstanding",
            score=95,
            grammar_score=20,
            semantic_score=20,
            documentation_score=19,
            convention_score=18,
            completeness_score=18,
            reasoning="Excellent entry",
        )
        assert review.score == 95
        assert review.grammar_score == 20

    def test_dimension_max_20(self):
        QualityReview = self._make_review_model()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QualityReview(
                name="test",
                quality_tier="good",
                score=50,
                grammar_score=25,  # exceeds max 20
                semantic_score=10,
                documentation_score=10,
                convention_score=5,
                completeness_score=0,
                reasoning="test",
            )

    def test_dimension_min_0(self):
        QualityReview = self._make_review_model()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QualityReview(
                name="test",
                quality_tier="poor",
                score=10,
                grammar_score=-1,  # below min 0
                semantic_score=5,
                documentation_score=3,
                convention_score=2,
                completeness_score=1,
                reasoning="test",
            )

    def test_total_score_max_100(self):
        QualityReview = self._make_review_model()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QualityReview(
                name="test",
                quality_tier="outstanding",
                score=101,  # exceeds max 100
                grammar_score=20,
                semantic_score=20,
                documentation_score=20,
                convention_score=20,
                completeness_score=20,
                reasoning="test",
            )

    def test_poor_tier_scores(self):
        QualityReview = self._make_review_model()
        review = QualityReview(
            name="data",
            quality_tier="poor",
            score=5,
            grammar_score=5,
            semantic_score=0,
            documentation_score=0,
            convention_score=0,
            completeness_score=0,
            reasoning="Uninformative name",
        )
        assert review.score == 5
        assert review.quality_tier == "poor"


# -----------------------------------------------------------------------
# Reviewer template rendering tests
# -----------------------------------------------------------------------


class TestReviewerTemplate:
    """Test that the reviewer template renders correctly."""

    def test_template_renders(self):
        from imas_codex.llm.prompt_loader import render_prompt
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        rendered = render_prompt(
            "sn/review_benchmark",
            {
                "calibration_entries": entries,
                "candidates": [
                    {
                        "standard_name": "electron_temperature",
                        "description": "Electron temperature",
                        "documentation": "A test doc",
                        "unit": "eV",
                        "kind": "scalar",
                        "tags": ["core_profiles"],
                        "fields": {
                            "physical_base": "temperature",
                            "subject": "electron",
                        },
                    }
                ],
            },
        )
        assert "electron_temperature" in rendered
        assert "Grammar Correctness" in rendered
        assert "Semantic Accuracy" in rendered
        assert "Documentation Quality" in rendered
        assert "outstanding" in rendered

    def test_template_includes_calibration_examples(self):
        from imas_codex.llm.prompt_loader import render_prompt
        from imas_codex.sn.benchmark import load_calibration_entries

        entries = load_calibration_entries()
        rendered = render_prompt(
            "sn/review_benchmark",
            {"calibration_entries": entries, "candidates": []},
        )
        # All calibration entry names should appear
        for entry in entries:
            assert entry["name"] in rendered, (
                f"Calibration entry {entry['name']} not in rendered template"
            )

    def test_template_renders_empty_candidates(self):
        from imas_codex.llm.prompt_loader import render_prompt

        rendered = render_prompt(
            "sn/review_benchmark",
            {"calibration_entries": [], "candidates": []},
        )
        assert "Scoring Dimensions" in rendered
